import abc
from typing import overload

from rdkit import Chem


class CustomOracle(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, mol: Chem.Mol) -> float: ...

    def evaluate_many(self, mols: list[Chem.Mol]) -> list[float]:
        return [self.evaluate(mol) for mol in mols]

    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: Chem.Mol | list[Chem.Mol]) -> float | list[float]:
        if isinstance(mol, Chem.Mol):
            return self.evaluate(mol)
        elif isinstance(mol, list):
            return self.evaluate_many(mol)
        else:
            raise TypeError("Input must be an RDKit Mol or a list of RDKit Mols.")
