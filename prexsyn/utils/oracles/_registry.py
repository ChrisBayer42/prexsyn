from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast, overload

import numpy as np
from rdkit import Chem


class OracleProtocol(Protocol):
    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...


_registry: dict[str, Callable[..., OracleProtocol]] = {}


_T = TypeVar("_T", bound=Callable[..., OracleProtocol])


def register(fn: _T) -> _T:
    _registry[fn.__name__] = fn
    return fn


def get_oracle(name: str, *args: Any, **kwargs: Any) -> OracleProtocol:
    if "+" in name or "*" in name:
        return ComposedOracle.from_string(name)

    if name not in _registry:
        raise ValueError(f"Unknown oracle '{name}'. Available oracles: {list(_registry.keys())}")
    return _registry[name](*args, **kwargs)


def has_oracle(name: str) -> bool:
    try:
        get_oracle(name)
        return True
    except ValueError:
        return False


class ComposedOracle:
    def __init__(self, *oracles: tuple[float, OracleProtocol]):
        self._funcs = [item[1] for item in oracles]
        self._weights = np.array([item[0] for item in oracles], dtype=np.float32)

    @classmethod
    def from_string(cls, s: str) -> "ComposedOracle":
        oracles: list[tuple[float, OracleProtocol]] = []
        items = [item.strip() for item in s.split("+")]
        for item in items:
            if "*" in item:
                weight_str, name = item.split("*")
                weight_str = weight_str.strip()
                name = name.strip()
                weight = float(weight_str)
            else:
                name = item
                weight = 1.0
            oracles.append((weight, get_oracle(name)))
        return cls(*oracles)

    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: Chem.Mol | list[Chem.Mol]) -> float | list[float]:
        if isinstance(mol, list):
            scores = np.array([fn(mol) for fn in self._funcs], dtype=np.float32)
            return cast(list[float], (scores * self._weights[:, None]).sum(axis=0).tolist())
        else:
            scores = np.array([fn(mol) for fn in self._funcs], dtype=np.float32)
            return float((scores * self._weights).sum())
