from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from prexsyn_engine.detokenizer import Detokenizer
from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef
from prexsyn_engine.synthesis import SynthesisVector


class Tokenization:
    def __init__(self, token_def: PostfixNotationTokenDef = PostfixNotationTokenDef()) -> None:
        self._token_def = token_def

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Tokenization":
        token_def = PostfixNotationTokenDef(**cfg.get("def", {}))
        return cls(token_def=token_def)

    @property
    def token_def(self) -> PostfixNotationTokenDef:
        return self._token_def


class DetokenizerWrapper:
    def __init__(self, detokenizer: Detokenizer) -> None:
        self._detokenizer = detokenizer

    def __call__(
        self,
        token_types: torch.Tensor | np.ndarray[Any, Any],
        bb_indices: torch.Tensor | np.ndarray[Any, Any],
        rxn_indices: torch.Tensor | np.ndarray[Any, Any],
    ) -> SynthesisVector:
        token_types = token_types.cpu().numpy() if isinstance(token_types, torch.Tensor) else token_types
        bb_indices = bb_indices.cpu().numpy() if isinstance(bb_indices, torch.Tensor) else bb_indices
        rxn_indices = rxn_indices.cpu().numpy() if isinstance(rxn_indices, torch.Tensor) else rxn_indices
        return self._detokenizer(
            token_types=token_types,
            bb_indices=bb_indices,
            rxn_indices=rxn_indices,
        )
