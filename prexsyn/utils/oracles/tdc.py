import pathlib
import pickle
from typing import Any, overload

import numpy as np
import requests  # type: ignore[import-untyped]
import sklearn.svm
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from ._registry import register


def _download(remote: str, local: str | pathlib.Path) -> None:
    response = requests.get(remote)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        with open(local, "wb") as file:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                file.write(data)

    if total_size != 0 and pbar.n != total_size:
        raise RuntimeError(f"Failed to download file from: {remote}")


@register
class drd2:
    model_url: str = "https://dataverse.harvard.edu/api/access/datafile/6413411"

    def __init__(self, model_path: str | pathlib.Path = "./data/oracles/drd2_current.pkl"):
        super().__init__()
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            _download(self.model_url, model_path)

        with open(model_path, "rb") as f:
            self.model: sklearn.svm.SVC = pickle.load(f)

    @staticmethod
    def _fingerprints_from_mol(mol: Chem.Mol) -> np.ndarray[Any, Any]:
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)  # type: ignore[attr-defined]
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            nfp[0, nidx] += int(v)
        return nfp

    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: list[Chem.Mol] | Chem.Mol) -> list[float] | float:
        if isinstance(mol, list):
            fp = np.concatenate([self._fingerprints_from_mol(m) for m in mol], axis=0)
        else:
            fp = self._fingerprints_from_mol(mol)

        score = self.model.predict_proba(fp)[:, 1]
        if isinstance(mol, list):
            return [float(s) for s in score]
        return float(score)
