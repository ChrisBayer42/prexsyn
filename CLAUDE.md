# PrexSyn — Claude Code Context

## Running the app
- Launch webapp: `./launch_webapp.sh` (from repo root) — starts Streamlit on port 8501 and opens Chrome
- Launch Jupyter: `./launch_jupyter.sh`
- Conda environment: `prexsyn` — use `conda run -n prexsyn <cmd>` or activate first
- Runtime is **conda**, not pixi (pixi manages the lock file but the live env is conda)

## Key files
- `prexsyn_webapp.py` — Streamlit webapp (~630 lines)
- `PrexSyn_Interactive_Exploration.ipynb` — interactive notebook
- `prexsyn/models/prexsyn.py` — transformer model
- `prexsyn/utils/draw.py` — molecule/synthesis drawing utilities
- `prexsyn/factories/load_model.py` — auto-downloads model from HuggingFace if missing
- `prexsyn/applications/analog/analog.py` — `generate_analogs()`
- `data/trained_models/v1_converted.yaml` — model config (required for webapp)

## Gotchas
- `st_ketcher()` first param is `value=`, NOT `molecule=` — wrong name raises TypeError on every render
- `st.form` is incompatible with `st_ketcher` — the Analog Generation page intentionally has no form wrapper
- `synthesis.top().to_list()` can return mixture SMILES (containing `.`) — always filter these before display
- Always call `AllChem.Compute2DCoords(mol)` before displaying molecules from synthesis ops; `draw_molecule()` has `recompute_coords=False` by default
- `draw.py` sets all atoms black via `setAtomPalette({0: (0,0,0)})` — deliberate design choice, do not change
- `load_model()` auto-downloads model + chemical space data from HuggingFace on first run

## Webapp session state keys (Analog Generation page)
- `active_input` — `"draw"` or `"text"`, determines which input is authoritative
- `draw_smiles` — last SMILES returned by Ketcher
- `text_smiles` — last value from the text input
- `example_smiles` — set by Home page example buttons, seeds both tabs
- `results` — dict with `input_smiles`, `input_mol`, `analogs`

## Security notes
- `seh_proxy.py:206` — deserialisation of raw HTTP stream via standard ML model loading; accepted risk, `# nosec` present
- `autodock.py:48` — XML parser warning is a false positive (requires Python >=3.11, issue fixed in 3.11)
