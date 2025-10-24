from typing import overload

from guacamol import scoring_function, standard_benchmarks
from rdkit import Chem

from ._registry import OracleProtocol, register


def scoring_function_wrapper(scoring_fn: scoring_function.ScoringFunction) -> OracleProtocol:
    @overload
    def wrapper(mol: Chem.Mol) -> float: ...
    @overload
    def wrapper(mol: list[Chem.Mol]) -> list[float]: ...

    def wrapper(mol: Chem.Mol | list[Chem.Mol]) -> list[float] | float:
        if isinstance(mol, list):
            smiles = [Chem.MolToSmiles(m) for m in mol]
            return [float(scoring_fn.score(s)) for s in smiles]
        else:
            smi = Chem.MolToSmiles(mol)
            return float(scoring_fn.score(smi))

    return wrapper


@register
def cobimetinib() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.hard_cobimetinib().objective)


@register
def osimertinib() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.hard_osimertinib().objective)


@register
def fexofenadine() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.hard_fexofenadine().objective)


@register
def perindopril() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.perindopril_rings().objective)


@register
def amlodipine() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.amlodipine_rings().objective)


@register
def ranolazine() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.ranolazine_mpo().objective)


@register
def sitagliptin() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.sitagliptin_replacement().objective)


@register
def zaleplon() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.zaleplon_with_other_formula().objective)


@register
def celecoxib_rediscovery() -> OracleProtocol:
    return scoring_function_wrapper(
        standard_benchmarks.similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ).objective
    )


@register
def guacamol_scaffold_hop() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.scaffold_hop().objective)


@register
def guacamol_decoration_hop() -> OracleProtocol:
    return scoring_function_wrapper(standard_benchmarks.decoration_hop().objective)
