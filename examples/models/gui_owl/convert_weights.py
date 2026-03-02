# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict

import torch
from executorch.examples.models.checkpoint import get_mapped_key

from executorch.examples.models.smollm3.convert_weights import load_checkpoint

# Weight mapping from HuggingFace Qwen3VL format to Meta format
# Based on Qwen3VLForConditionalGeneration architecture
_QWEN3VL_TO_META = {
    # Text decoder weights
    "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
    "language_model.model.norm.weight": "norm.weight",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def qwen3vl_tune_to_meta(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HuggingFace Qwen3VL format to Meta format.
    This function extracts only the text decoder weights, following the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HuggingFace Qwen3VL format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta format for text decoder.
    """
    converted_text_model_state_dict = {}
    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _QWEN3VL_TO_META)
            converted_text_model_state_dict[new_key] = value
        except:
            # Only preserve parameters of text decoder, skip vision encoder and projector
            pass

    # Handle tied embeddings - if no lm_head, tie to token embeddings
    if "output.weight" not in converted_text_model_state_dict:
        if "language_model.lm_head.weight" in state_dict:
            converted_text_model_state_dict["output.weight"] = state_dict[
                "language_model.lm_head.weight"
            ]
        else:
            # Tied embeddings
            converted_text_model_state_dict["output.weight"] = converted_text_model_state_dict[
                "tok_embeddings.weight"
            ]

    return converted_text_model_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    converted_sd = qwen3vl_tune_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(converted_sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-Owl (Qwen3VL) weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
