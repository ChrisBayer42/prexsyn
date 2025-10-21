from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict, TypeVar, no_type_check

import torch

EmbedderName: TypeAlias = str
EmbedderParams: TypeAlias = Mapping[str, torch.Tensor]


class SynthesisRepr(TypedDict):
    token_types: torch.Tensor
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor


PropertyRepr: TypeAlias = Sequence[Mapping[EmbedderName, EmbedderParams]] | Mapping[EmbedderName, EmbedderParams]


class SynthesisTrainingBatch(TypedDict):
    synthesis_repr: SynthesisRepr
    property_repr: PropertyRepr


_T = TypeVar("_T")


@no_type_check
def move_to_device(d: _T, device: torch.device | str) -> _T:
    if isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, Mapping):
        return {k: move_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, Sequence):
        return [move_to_device(item, device) for item in d]
    else:
        return d
