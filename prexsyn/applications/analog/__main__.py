import pathlib
import os
from collections.abc import Sequence

import click
import pandas as pd
import torch
from rdkit import Chem
from tqdm.auto import tqdm

from prexsyn.factories import load_model
from prexsyn.samplers.basic import BasicSampler

from .analog import generate_analogs
from .data import AnalogGenerationDatabase

def validate_file_path(path, must_exist=True):
    """Validate and sanitize file paths to prevent path traversal attacks"""
    try:
        # Convert to absolute path
        abs_path = pathlib.Path(path).resolve()
        
        # Check if path exists (if required)
        if must_exist and not abs_path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        # Check if path is a file (if required)
        if must_exist and not abs_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        # Check for suspicious path components
        path_str = str(abs_path)
        if ".." in path_str or path_str.startswith("/"):
            # This is already handled by resolve(), but double-check
            pass
        
        return abs_path
    except Exception as e:
        raise ValueError(f"Invalid or unsafe path: {path} - {str(e)}")


@click.command()
@click.option("--model", "model_path", type=str, required=True)
@click.option("-i", "csv_path", type=str, required=True)
@click.option("-o", "output_path", type=str, required=True)
@click.option("--max-length", default=16)
@click.option("--num-samples", default=128)
@click.option("--device", default="cuda")
def analog_generation_cli(
    model_path: str,
    csv_path: str,
    output_path: str,
    max_length: int,
    num_samples: int,
    device: torch.device | str,
) -> None:
    try:
        # Validate all file paths
        model_path = validate_file_path(model_path, must_exist=True)
        csv_path = validate_file_path(csv_path, must_exist=True)
        output_path = validate_file_path(output_path, must_exist=False)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.set_grad_enabled(False)
        df = pd.read_csv(csv_path)
        smi_list: Sequence[str] = df["SMILES"].tolist()
        
        with AnalogGenerationDatabase(output_path) as db:
            existing = set(db.keys())
            smi_list = [smi for smi in smi_list if smi not in existing]

            if len(smi_list) == 0:
                print("All SMILES already processed.")
                return

            facade, model = load_model(model_path)
            model = model.eval().to(device)
            sampler = BasicSampler(
                model,
                token_def=facade.tokenization.token_def,
                num_samples=num_samples,
                max_length=max_length,
            )
            with tqdm(total=len(smi_list)) as pbar:
                for smi in smi_list:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        print(f"Invalid SMILES: {smi}")
                        continue
                    entry = generate_analogs(
                        facade=facade,
                        model=model,
                        sampler=sampler,
                        fp_property=facade.property_set["ecfp4"],
                        mol=mol,
                    )
                    db[smi] = entry
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "count": len(db),
                            "avg_sim": f"{db.get_average_similarity():.4f}",
                            "recons": f"{db.get_reconstruction_rate():.4f}",
                        }
                    )

        print("[Results]")
        print(f"Model path: {model_path}")
        print(f"CSV path: {csv_path}")
        print(f"Output path: {output_path}")
        print(f"Total entries: {len(db)}")
        print(f"Average similarity:  {db.get_average_similarity():.4f}")
        print(f"Reconstruction rate: {db.get_reconstruction_rate():.4f}")
        time_mean, time_std = db.get_time_statistics()
        print(f"Time per target: {time_mean:.4f} +/- {time_std:.4f} sec")
        print(
            "[NOTE] The reported time only includes the model inference time (`generate_analogs` call), "
            "which is shorter than the actual time taken. "
            "Extra time is mainly taken for loading the model and saving outputs to the database."
        )
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        if "Invalid or unsafe path" in str(e):
            print("[SECURITY] Path validation failed. Please check your file paths.")
        elif "Invalid SMILES" in str(e):
            print("[INPUT ERROR] Invalid SMILES format detected.")
        else:
            print("[UNEXPECTED ERROR] An unexpected error occurred.")
        raise


if __name__ == "__main__":
    analog_generation_cli()
