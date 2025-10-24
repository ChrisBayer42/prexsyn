from collections.abc import Callable
from typing import cast, overload

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

from ._registry import OracleProtocol, register


def oracle_function_wrapper(fn: Callable[[Chem.Mol], float]) -> OracleProtocol:
    @overload
    def wrapper(mol: Chem.Mol) -> float: ...
    @overload
    def wrapper(mol: list[Chem.Mol]) -> list[float]: ...

    def wrapper(mol: Chem.Mol | list[Chem.Mol]) -> list[float] | float:
        if isinstance(mol, list):
            return [float(fn(m)) for m in mol]
        else:
            return float(fn(mol))

    return wrapper


@register
def qed() -> OracleProtocol:
    return oracle_function_wrapper(lambda m: QED.qed(m))  # type: ignore[no-untyped-call]


def _soft_less_than(x: float, target: float, inv_lambda: float) -> float:
    diff = x - target
    if diff < 0:
        diff = 0
    return float(np.exp(-inv_lambda * diff))


def _lipinski(mol: Chem.Mol) -> list[float]:
    rule_1 = _soft_less_than(Descriptors.ExactMolWt(mol), 500 - 10, 10)  # type: ignore[attr-defined]
    rule_2_don = _soft_less_than(Lipinski.NumHDonors(mol), 5 - 1, 1)  # type: ignore[attr-defined]
    rule_2_acc = _soft_less_than(Lipinski.NumHAcceptors(mol), 10 - 1, 1)  # type: ignore[attr-defined]
    rule_3 = _soft_less_than(Descriptors.TPSA(mol), 140 - 10, 10)  # type: ignore[attr-defined]
    rule_4 = _soft_less_than(Crippen.MolLogP(mol), 5 - 0.5, 1)  # type: ignore[attr-defined]
    rule_5 = _soft_less_than(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol), 10 - 1, 1)
    return [rule_1, rule_2_don, rule_2_acc, rule_3, rule_4, rule_5]


@register
def lipinski() -> OracleProtocol:
    def _fn(mol: Chem.Mol) -> float:
        scores = _lipinski(mol)
        return float(np.mean(scores))

    return oracle_function_wrapper(_fn)


@register
def lipinski_product() -> OracleProtocol:
    def _fn(mol: Chem.Mol) -> float:
        scores = _lipinski(mol)
        prod = 1.0
        for s in scores:
            prod *= s
        return prod

    return oracle_function_wrapper(_fn)


@register
def scaffold_hop_demo1() -> OracleProtocol:
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    from rdkit.DataStructs import TanimotoSimilarity

    ref_mol = Chem.MolFromSmiles("CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C")
    ref_ph_fp = Generate.Gen2DFingerprint(ref_mol, Gobbi_Pharm2D.factory)  # type: ignore[no-untyped-call]
    deco1 = Chem.MolFromSmiles("c1ccc2ncsc2c1")
    deco2 = Chem.MolFromSmiles("CCCO")
    core = Chem.MolFromSmiles("[#7]-c1ncnc2cc(-[#8])ccc12")
    core_bitset = set(cast(list[int], np.array(Chem.LayeredFingerprint(core)).nonzero()[0].tolist()))

    def _score_fn(mol: Chem.Mol) -> float:
        mol_bitset = set(cast(list[int], np.array(Chem.LayeredFingerprint(mol)).nonzero()[0].tolist()))

        contains_deco1 = 1.0 if mol.HasSubstructMatch(deco1) else 0.0
        contains_deco2 = 1.0 if mol.HasSubstructMatch(deco2) else 0.0

        not_contains_core = 1.0 - len(mol_bitset.intersection(core_bitset)) / len(core_bitset)

        mol_ph_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)  # type: ignore[no-untyped-call]
        sim_to_ref: float = TanimotoSimilarity(mol_ph_fp, ref_ph_fp)

        return (contains_deco1 + contains_deco2 + not_contains_core + sim_to_ref) / 4

    return oracle_function_wrapper(_score_fn)
