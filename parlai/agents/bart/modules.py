#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART Module.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Any, Dict, Union, List, Optional

from parlai.agents.transformer.modules import TransformerGeneratorModel


class BartModel(TransformerGeneratorModel):
    """
    BART Model.
    """

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits.

        Override standard TGM output to _not_ prevent generation of BOS.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output

    def _get_initial_forced_decoder_input(self, bsz: int, inputs: torch.LongTensor):
        """
        Return initial input to the decoder.

        :param bsz:
            batchsize
        :param inputs:
            inputs to decode

        :return initial_input:
            initial input for the decoder.
        """
        return torch.cat([torch.LongTensor([self.END_IDX]).to(inputs).detach().expand(bsz, 1), inputs], 1)

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[str, Any],
        inds: Union[List[int], torch.LongTensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Incremental state is too hard with all the docs and what not.

        We leave as a future exercise.
        """
        return None