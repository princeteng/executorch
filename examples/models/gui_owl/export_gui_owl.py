# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export script for GUI-Owl-1.5-2B-Instruct model.

GUI-Owl is a vision-language model based on Qwen3VL architecture.
This script exports the model to ExecuTorch .pte format.

Usage:
    python -m executorch.examples.models.gui_owl.export_gui_owl \
        --max-context-len 768 \
        --max-seq-len 768 \
        --pte-name gui_owl_xnnpack.pte
"""

import logging
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.examples.models.llama.export_llama_lib import (
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
    get_quant_weight_transform,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from executorch.extension.llm.export.config.llm_config import LlmConfig
from executorch.util.activation_memory_profiler import generate_memory_trace
from pytorch_tokenizers.llama2c import Llama2cTokenizer as Tokenizer
from torch.export import Dim
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class GUIOwlEdgeManager(LLMEdgeManager):
    def export(self) -> "GUIOwlEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


def export_text_model(gui_owl, embeddings, dynamic_shapes):
    class GUIOwlTextModel(torch.nn.Module):
        """Text decoder model for GUI-Owl."""

        def __init__(self, gui_owl):
            super().__init__()
            self.text_model = gui_owl.text_model

        def forward(self, embeddings, input_pos):
            return self.text_model(None, {"input_pos": input_pos}, embeddings)

    gui_owl_text_model = GUIOwlTextModel(gui_owl)

    text_model_em = LLMEdgeManager(
        model=gui_owl_text_model,
        modelname="gui_owl_text_model",
        max_seq_len=gui_owl.text_model_args.max_seq_len,
        dtype=DType.fp32,
        use_kv_cache=True,
        example_inputs=(embeddings, torch.tensor([0], dtype=torch.int64)),
        dynamic_shapes=dynamic_shapes,
    )

    llm_config = LlmConfig()
    llm_config.base.params = "params.json"
    llm_config.backend.xnnpack.enabled = True
    llm_config.quantization.qmode = "8da4w"
    llm_config.quantization.group_size = 128
    llm_config.quantization.embedding_quantize = "4,32"

    dtype_override = DType.fp32
    quant_transform = get_quant_weight_transform(
        quantization_mode=llm_config.quantization.qmode,
        group_size=llm_config.quantization.group_size,
        computation_dtype=dtype_override,
        checkpoint_path=llm_config.base.checkpoint,
        tokenizer_path=llm_config.base.tokenizer_path,
        calibration_tasks=llm_config.quantization.calibration_tasks,
        calibration_limit=llm_config.quantization.calibration_limit,
        calibration_seq_length=llm_config.quantization.calibration_seq_length,
    )
    _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
    source_transforms = []
    if gui_owl.use_sdpa_with_kv_cache_op:
        source_transforms.append(replace_kv_cache_with_custom_kv_cache)
        source_transforms.append(replace_sdpa_with_custom_op)
    source_transforms.append(quant_transform)
    manager = (
        text_model_em.set_output_dir("./")
        .to_dtype(dtype_override)
        .source_transform(source_transforms)
        .export()
        .pt2e_quantize(quantizers)
    )

    with torch.no_grad():
        text_model_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager._get_dynamic_shape(),
            strict=True,
        )
    return text_model_ep


def export_image_encoder(gui_owl, resized, dynamic_shapes):
    class GUIOwlImageEncoder(torch.nn.Module):
        """Image encoder model for GUI-Owl."""

        def __init__(self, gui_owl):
            super().__init__()
            self.gui_owl = gui_owl

        def forward(self, images):
            return self.gui_owl.image_embedding(images)

    gui_owl_image_encode = GUIOwlImageEncoder(gui_owl)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    manager = (
        GUIOwlEdgeManager(
            model=gui_owl_image_encode,
            modelname="gui_owl_image_encoder",
            max_seq_len=gui_owl.text_model_args.max_seq_len,
            dtype=DType.fp32,
            use_kv_cache=True,
            example_inputs=(resized,),
            dynamic_shapes=dynamic_shapes,
        )
        .export()
        .pt2e_quantize([quantizer])
    )

    with torch.no_grad():
        image_encoder_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager.dynamic_shapes,
            strict=True,
        )
    return image_encoder_ep


def export_token_embedding(gui_owl, prompt):
    def quant_embedding(model):
        return EmbeddingQuantHandler(
            model,
            bitwidth=8,
            group_size=32,
            packed=False,
        ).quantized_model()

    quantized_token_embed = quant_embedding(gui_owl.model_.model.language_model)
    token_dim_1 = Dim("token_dim_1", min=2, max=gui_owl.text_model_args.max_seq_len)
    dynamic_shapes = [{1: token_dim_1}]
    with torch.no_grad():
        token_embedding_ep = torch.export.export(
            quantized_token_embed.embed_tokens,
            (prompt,),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    return token_embedding_ep


def export_all(gui_owl_model):
    gui_owl = gui_owl_model.get_eager_model()

    (
        prompt_before_image,
        resized,
        prompt_after_image,
    ) = gui_owl_model.get_inputs_for_prefill()

    image_encoder_ep = export_image_encoder(
        gui_owl, resized, gui_owl_model._get_image_dynamic_shapes()
    )

    embeddings = gui_owl.prefill_embedding(
        prompt_before_image, resized, prompt_after_image
    )

    text_model_ep = export_text_model(
        gui_owl, embeddings, gui_owl_model._get_prompt_dynamic_shapes()
    )

    token_embedding_ep = export_token_embedding(gui_owl, prompt_before_image)

    lowered_and_edge = to_edge_transform_and_lower(
        {
            "vision_encoder": image_encoder_ep,
            "token_embedding": token_embedding_ep,
            "text_decoder": text_model_ep,
        },
        partitioner={
            "vision_encoder": [XnnpackPartitioner()],
            "text_decoder": [
                XnnpackPartitioner(
                    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                    per_op_mode=True,
                ),
                XnnpackPartitioner(),
            ],
        },
        constant_methods={"get_max_seq_len": gui_owl_model.max_seq_len},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    executorch_program = lowered_and_edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": HintBasedSymShapeEvalPass(),
            },
        )
    )
    for execution_plan in executorch_program._emitter_output.program.execution_plan:
        logging.info(
            f"Required memory for activation in bytes: {execution_plan.non_const_buffer_sizes}"
        )
    return executorch_program


def get_tokenizer_for_gui_owl_runner(gui_owl_model):
    gui_owl_model.tokenizer.save_vocabulary("./")
    t = Tokenizer("tokenizer.model")
    t.export("tokenizer.bin")


def create_gui_owl_config_from_args(args):
    llm_config = LlmConfig()
    llm_config.model.use_sdpa_with_kv_cache = args.use_sdpa_with_kv_cache
    llm_config.export.max_context_length = args.max_context_len
    llm_config.export.max_seq_length = args.max_seq_len
    llm_config.export.output_name = args.pte_name
    llm_config.debug.profile_memory = args.profile_memory
    return llm_config


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--use-sdpa-with-kv-cache",
        default=True,
        action=BooleanOptionalAction,
        help="Use sdpa_with_kv_cache custom op in GUI-Owl text model.",
    )
    parser.add_argument(
        "--max-context-len",
        required=True,
        type=int,
        help="Maximum context length for the text model.",
    )
    parser.add_argument(
        "--max-seq-len",
        default=768,
        type=int,
        help="Maximum sequence length for the text model.",
    )
    parser.add_argument(
        "--pte-name",
        default="gui_owl_combined_xnnpack.pte",
        help="Name of the exported ExecuTorch program.",
    )
    parser.add_argument(
        "--with-artifacts",
        default=False,
        action=BooleanOptionalAction,
        help="Generate artifacts for GUI-Owl runner.",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    args = parser.parse_args()

    llm_config = create_gui_owl_config_from_args(args)

    logging.info(
        f"Exporting GUI-Owl model to ExecuTorch with sdpa_with_kv_cache: {llm_config.model.use_sdpa_with_kv_cache}, max_seq_len: {llm_config.export.max_seq_length}, max_context_len: {llm_config.export.max_context_length}"
    )

    # TODO: Implement GUIOwlModel class similar to LlavaModel
    # For now, this is a template for the export flow
    logging.warning(
        "GUI-Owl model export is a work in progress. "
        "You need to implement GUIOwlModel class and GUIOwl class "
        "similar to LlavaModel and Llava in llava/model.py"
    )

    # executorch_program = export_all(gui_owl_model)
    #
    # if llm_config.debug.profile_memory:
    #     for method_name in executorch_program.methods:
    #         generate_memory_trace(
    #             executorch_program,
    #             f"{llm_config.export.output_name}_{method_name}.json",
    #             method_name=method_name,
    #         )
    #
    # with open(llm_config.export.output_name, "wb") as f:
    #     executorch_program.write_to_file(f)
    # logging.info(f"Exported ExecuTorch program to {llm_config.export.output_name}")
    #
    # if args.with_artifacts:
    #     get_tokenizer_for_gui_owl_runner(gui_owl_model)


if __name__ == "__main__":
    main()
