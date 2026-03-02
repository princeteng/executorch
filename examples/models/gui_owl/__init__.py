# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.gui_owl.convert_weights import convert_weights
from executorch.examples.models.llama.model import Llama2Model


class Qwen3VLModel(Llama2Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "Qwen3VLModel",
    "convert_weights",
]
