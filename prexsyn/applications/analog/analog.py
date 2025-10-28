import time

from rdkit import Chem

from prexsyn.data.struct import move_to_device
from prexsyn.factories.facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.properties import BasePropertyDef
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.fingerprints import mol_to_syntheses_tanimoto_similarity

from .data import AnalogGenerationResult


def generate_analogs(
    facade: Facade,
    model: PrexSyn,
    sampler: BasicSampler,
    fp_property: BasePropertyDef,
    mol: Chem.Mol,
    eval_fp_type: str = "ecfp4",
) -> AnalogGenerationResult:
    t_start = time.perf_counter()
    property_repr = {fp_property.name: move_to_device(fp_property.evaluate_mol(mol), model.device)}
    synthesis_repr = sampler.sample(property_repr)
    syn_list = facade.get_detokenizer()(**synthesis_repr)

    sim_matrix = mol_to_syntheses_tanimoto_similarity(mol, syn_list, fp_type=eval_fp_type)
    sim_list = sim_matrix.max(axis=1)
    max_sim_product_idx = sim_matrix.argmax(axis=1)
    t_end = time.perf_counter()
    return {
        "synthesis": list(syn_list),
        "similarity": sim_list,
        "max_sim_product_idx": max_sim_product_idx,
        "time_taken": t_end - t_start,
    }
