#!/usr/bin/env python3
"""
PrexSyn Web Application
A professional interface for exploring synthesizable chemical space.
"""

import pathlib
import re
import tempfile

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import streamlit as st

from prexsyn.applications.analog import generate_analogs
from prexsyn.factories import load_model
from prexsyn.utils.draw import draw_synthesis, draw_molecule
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.synthesis import Synthesis
from prexsyn_engine.fingerprints import tanimoto_similarity
from streamlit_ketcher import st_ketcher

# ── Page configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrexSyn",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Typography */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

  /* Hide Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }

  /* App header banner */
  .app-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #4338CA 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.75rem;
  }
  .app-header h1 {
    font-size: 1.9rem; font-weight: 700; margin: 0 0 0.2rem 0;
    letter-spacing: -0.02em;
  }
  .app-header p { font-size: 0.9rem; margin: 0; opacity: 0.82; }

  /* Cards */
  .card {
    background: white; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 1.25rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .card h4 { font-size: 0.9rem; font-weight: 600; color: #1E293B; margin: 0 0 0.6rem 0; }

  /* Molecule display */
  .mol-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 0.5rem;
    text-align: center;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
  }
  .mol-card svg {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }

  /* Similarity badge */
  .sim-badge {
    display: inline-block; background: #EFF6FF; color: #1D4ED8;
    border: 1px solid #BFDBFE; font-size: 0.78rem; font-weight: 600;
    padding: 0.2rem 0.65rem; border-radius: 99px;
  }

  /* Property table */
  .prop-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  .prop-table td { padding: 0.38rem 0.5rem; border-bottom: 1px solid #F1F5F9; }
  .prop-table td:first-child { color: #64748B; width: 52%; }
  .prop-table td:last-child  { font-weight: 500; color: #1E293B; }

  /* Feature cards */
  .feature-card {
    background: white; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 1.25rem; height: 100%;
  }
  .feature-card .icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
  .feature-card h4 { font-size: 0.88rem; font-weight: 600; color: #1E293B; margin: 0 0 0.45rem 0; }
  .feature-card ul { font-size: 0.8rem; color: #475569; padding-left: 1.1rem; margin: 0; line-height: 1.65; }

  /* Sidebar branding */
  .sidebar-brand {
    text-align: center; padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;
  }
  .sidebar-brand h2 { font-size: 1.25rem; font-weight: 700; color: #1D4ED8; margin: 0 0 0.1rem 0; }
  .sidebar-brand p  { font-size: 0.72rem; color: #64748B; margin: 0; }

  /* Status dots */
  .status-ok   { color: #059669; font-weight: 500; font-size: 0.85rem; }
  .status-warn { color: #D97706; font-weight: 500; font-size: 0.85rem; }

  /* Button overrides */
  .stButton > button { border-radius: 8px; font-weight: 500; font-size: 0.875rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────────
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.facade = None
    st.session_state.model = None

# ── Model loading ────────────────────────────────────────────────────────────────
def load_prexsyn_model() -> bool:
    model_path = pathlib.Path("./data/trained_models/v1_converted.yaml")
    if not model_path.exists():
        st.error("Model file not found at `data/trained_models/v1_converted.yaml`.")
        return False
    try:
        with st.spinner("Loading model…"):
            facade, model = load_model(model_path, train=False)
            model = model.to("cpu")
            st.session_state.facade = facade
            st.session_state.model = model
            st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return False

# ── Molecule helpers ──────────────────────────────────────────────────────────────
def render_mol_svg(smiles: str, size: tuple[int, int] = (300, 240)) -> str | None:
    """
    Render a SMILES string as a polished SVG image string, or None on failure.

    Drawing choices:
    - Explicit 2D coordinate computation ensures clean layouts for molecules
      that originate from synthesis operations and may lack coordinates.
    - PrepareMolForDrawing normalises wedge bonds, kekulises aromatic systems,
      and adds stereo annotations so the image matches chemical convention.
    - CPK colouring (default palette) is kept so heteroatoms are immediately
      identifiable; this is more informative than a monochrome style.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        rdMolDraw2D.PrepareMolForDrawing(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.setBackgroundColour((1, 1, 1, 0))   # transparent — card CSS provides bg
        opts.bondLineWidth = 2.0
        opts.addStereoAnnotation = True           # show R/S and E/Z labels
        opts.additionalAtomLabelPadding = 0.1
        opts.padding = 0.12                       # breathing room around the structure

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def mol_properties(mol: Chem.Mol) -> dict:
    return {
        "Formula":    CalcMolFormula(mol),
        "MW (g/mol)": f"{Descriptors.MolWt(mol):.2f}",
        "LogP":       f"{Descriptors.MolLogP(mol):.2f}",
        "HBA":        Descriptors.NumHAcceptors(mol),
        "HBD":        Descriptors.NumHDonors(mol),
        "Rot. bonds": Descriptors.NumRotatableBonds(mol),
        "TPSA (Å²)":  f"{Descriptors.TPSA(mol):.1f}",
    }


def lipinski_pass(mol: Chem.Mol) -> bool:
    return (
        Descriptors.MolWt(mol) <= 500
        and Descriptors.MolLogP(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 10
        and Descriptors.NumHDonors(mol) <= 5
    )


def validate_smiles(smiles: str) -> bool:
    """Return True if smiles is a valid, parseable SMILES string."""
    if not smiles or len(smiles) > 2000:
        return False
    # Reject characters that cannot appear in any SMILES but could be injections
    forbidden = {";", "&", "|", "<", ">", "`", "\n", "\r"}
    if any(c in smiles for c in forbidden):
        return False
    return Chem.MolFromSmiles(smiles) is not None


def is_mixture(smiles: str) -> bool:
    """Return True if the SMILES represents disconnected fragments."""
    return "." in smiles


# ── Analog generation ────────────────────────────────────────────────────────────
def process_molecule(smiles: str, num_results: int, num_samples: int) -> None:
    if not validate_smiles(smiles):
        st.error("Invalid SMILES string.")
        return

    mol = Chem.MolFromSmiles(smiles)
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)

    sampler = BasicSampler(
        st.session_state.model,
        token_def=st.session_state.facade.tokenization.token_def,
        num_samples=num_samples,
        max_length=16,
    )

    try:
        result = generate_analogs(
            facade=st.session_state.facade,
            model=st.session_state.model,
            sampler=sampler,
            fp_property=st.session_state.facade.property_set["ecfp4"],
            mol=mol,
        )
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return

    visited: set[str] = set()
    result_list = []
    for synthesis in result["synthesis"]:
        if synthesis.stack_size() != 1:
            continue
        for prod in synthesis.top().to_list():
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            if is_mixture(prod_smi) or prod_smi in visited:
                continue
            visited.add(prod_smi)
            sim = tanimoto_similarity(prod, mol, fp_type="ecfp4")
            result_list.append((prod, synthesis, sim))

    result_list.sort(key=lambda x: x[2], reverse=True)

    # Find a synthesis that reconstructs the input molecule itself (often present)
    input_synthesis = None
    for synthesis in result["synthesis"]:
        if synthesis.stack_size() != 1:
            continue
        for prod in synthesis.top().to_list():
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            if prod_smi == canonical_smi and not is_mixture(prod_smi):
                input_synthesis = synthesis
                break
        if input_synthesis:
            break

    st.session_state["results"] = {
        "input_smiles":     canonical_smi,
        "input_mol":        mol,
        "results":          result_list[:num_results],
        "input_synthesis":  input_synthesis,
    }


# ── Pages ────────────────────────────────────────────────────────────────────────
def show_home_page() -> None:
    st.markdown("""
    <div class="app-header">
        <h1>⚗ PrexSyn</h1>
        <p>Efficient, programmable exploration of synthesizable chemical space</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("⚗", "Analog Generation", [
            "Find synthesizable analogs of any query molecule",
            "Results ranked by Tanimoto similarity",
            "Fully automated — no retrosynthesis expertise required",
        ]),
        ("⬡", "Synthesis Pathways", [
            "Step-by-step building block + reaction trees",
            "Based on real Enamine building blocks",
            "Visual pathway diagrams with intermediates",
        ]),
        ("⊞", "Batch Processing", [
            "Validate and inspect multiple molecules at once",
            "Molecular property summary for each entry",
            "One-click jump to analog generation",
        ]),
    ]
    for col, (icon, title, points) in zip([c1, c2, c3], features):
        with col:
            pts = "".join(f"<li>{p}</li>" for p in points)
            st.markdown(f"""
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <ul>{pts}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example molecules — click to explore:**")
    examples = {
        "Aspirin":   "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine":  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Benzene":   "C1=CC=CC=C1",
        "Ethanol":   "CCO",
    }
    cols = st.columns(len(examples))
    for col, (name, smiles) in zip(cols, examples.items()):
        with col:
            if st.button(name, key=f"home_{name}", use_container_width=True):
                st.session_state["example_smiles"] = smiles
                st.session_state["goto_page"] = "Analog Generation"
                st.rerun()


def show_analog_generation(num_results: int, num_samples: int) -> None:
    st.subheader("Analog Generation")
    st.caption(
        "Draw a molecule or enter a SMILES string, then click Generate Analogs. "
        "Results are ranked by Tanimoto similarity to the query molecule."
    )

    # ── Input section ────────────────────────────────────────────────────────
    input_col, preview_col = st.columns([3, 2], gap="large")

    with input_col:
        tab_draw, tab_text = st.tabs(["✏ Draw", "⌨ SMILES"])

        # ── Draw tab ─────────────────────────────────────────────────────────
        with tab_draw:
            # Pre-populate Ketcher with whatever SMILES is already active
            seed_smiles = st.session_state.get(
                "draw_smiles",
                st.session_state.get("example_smiles", ""),
            )
            drawn = st_ketcher(value=seed_smiles, height=500, key="ketcher")

            # Detect if the user has drawn something new
            if drawn and drawn != st.session_state.get("draw_smiles", ""):
                st.session_state["draw_smiles"] = drawn
                st.session_state["active_input"] = "draw"

            # Show the resulting SMILES so the user can inspect / copy it
            current_drawn = st.session_state.get("draw_smiles", "")
            if current_drawn:
                st.code(current_drawn, language=None)

        # ── Text tab ─────────────────────────────────────────────────────────
        with tab_text:
            def _mark_text_active() -> None:
                st.session_state["active_input"] = "text"

            text_smiles = st.text_input(
                "SMILES",
                value=st.session_state.get(
                    "text_smiles",
                    st.session_state.get("example_smiles", "CC(=O)OC1=CC=CC=C1C(=O)O"),
                ),
                help="Example: CC(=O)OC1=CC=CC=C1C(=O)O  (Aspirin)",
                key="text_smiles_input",
                on_change=_mark_text_active,
            )
            st.session_state["text_smiles"] = text_smiles

    # ── Resolve active SMILES ─────────────────────────────────────────────────
    active_input = st.session_state.get("active_input", "text")
    if active_input == "draw":
        active_smiles = st.session_state.get("draw_smiles", "")
    else:
        active_smiles = st.session_state.get("text_smiles", "CC(=O)OC1=CC=CC=C1C(=O)O")

    mol_preview = Chem.MolFromSmiles(active_smiles) if active_smiles else None

    # ── Live preview (right column) ───────────────────────────────────────────
    with preview_col:
        st.markdown("**Preview**")
        if mol_preview:
            svg = render_mol_svg(active_smiles)
            if svg:
                st.markdown(f'<div class="mol-card">{svg}</div>', unsafe_allow_html=True)
            props = mol_properties(mol_preview)
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in props.items()
            )
            lp = lipinski_pass(mol_preview)
            lp_cell = (
                '<span style="color:#059669;font-weight:600">Pass</span>'
                if lp else
                '<span style="color:#DC2626;font-weight:600">Fail</span>'
            )
            st.markdown(
                f'<table class="prop-table">{rows}'
                f'<tr><td>Lipinski Ro5</td><td>{lp_cell}</td></tr></table>',
                unsafe_allow_html=True,
            )
        elif active_smiles:
            st.warning("Invalid SMILES — please check the input.")
        else:
            st.info("Draw or enter a molecule to see a preview.")

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_col, clr_col = st.columns([3, 1])
    with btn_col:
        generate = st.button(
            "Generate Analogs",
            type="primary",
            use_container_width=True,
            disabled=(mol_preview is None),
        )
    with clr_col:
        if st.button("Clear", use_container_width=True):
            for key in ("results", "draw_smiles", "text_smiles", "example_smiles",
                        "active_input"):
                st.session_state.pop(key, None)
            st.rerun()

    if generate:
        with st.spinner("Generating analogs…"):
            process_molecule(active_smiles, num_results, num_samples)

    if st.session_state.get("results"):
        _display_results(st.session_state["results"])


def _display_results(results_data: dict) -> None:
    st.markdown("---")
    st.subheader("Input Molecule")

    in_smi = results_data["input_smiles"]
    in_mol = results_data["input_mol"]
    ic1, ic2 = st.columns([1, 2])
    with ic1:
        svg = render_mol_svg(in_smi)
        if svg:
            st.markdown(f'<div class="mol-card">{svg}</div>', unsafe_allow_html=True)
    with ic2:
        props = mol_properties(in_mol)
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in props.items())
        st.markdown(f'<table class="prop-table">{rows}</table>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.8rem;font-family:monospace;color:#475569;margin-top:0.6rem">'
            f'{in_smi}</p>',
            unsafe_allow_html=True,
        )
        if results_data.get("input_synthesis"):
            if st.button("View synthesis pathway", key="synth_input"):
                st.session_state["synthesis_to_show"] = results_data["input_synthesis"]
                st.session_state["goto_page"] = "Synthesis Visualization"
                st.rerun()

    results = results_data["results"]
    if not results:
        st.info("No valid analogs found. Try increasing the number of samples.")
        return

    st.subheader(f"Top {len(results)} Analog{'s' if len(results) != 1 else ''}")

    for i, (prod, synthesis, sim) in enumerate(results):
        prod_smi = Chem.MolToSmiles(prod, canonical=True)
        with st.expander(
            f"#{i+1}  ·  {CalcMolFormula(prod)}  ·  similarity {sim:.3f}",
            expanded=(i == 0),
        ):
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                svg = render_mol_svg(prod_smi)
                if svg:
                    st.markdown(f'<div class="mol-card">{svg}</div>', unsafe_allow_html=True)
            with rc2:
                props = mol_properties(prod)
                rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in props.items())
                lp = lipinski_pass(prod)
                lp_cell = (
                    '<span style="color:#059669;font-weight:600">Pass</span>'
                    if lp else
                    '<span style="color:#DC2626;font-weight:600">Fail</span>'
                )
                st.markdown(
                    f'<span class="sim-badge">Similarity {sim:.3f}</span>'
                    f'<table class="prop-table" style="margin-top:0.65rem">{rows}'
                    f'<tr><td>Lipinski Ro5</td><td>{lp_cell}</td></tr></table>'
                    f'<p style="font-size:0.78rem;font-family:monospace;color:#64748B;'
                    f'word-break:break-all;margin-top:0.5rem">{prod_smi}</p>',
                    unsafe_allow_html=True,
                )
                if st.button("View synthesis pathway", key=f"synth_{i}"):
                    st.session_state["synthesis_to_show"] = synthesis
                    st.session_state["goto_page"] = "Synthesis Visualization"
                    st.rerun()


def show_synthesis_visualization(num_results: int, num_samples: int) -> None:
    st.subheader("Synthesis Pathway")
    st.caption(
        "Step-by-step visualization of how the selected molecule is assembled "
        "from building blocks using known reaction templates."
    )

    if not st.session_state.get("synthesis_to_show"):
        st.info(
            "No synthesis selected. Go to **Analog Generation**, generate analogs, "
            "then click **View synthesis pathway** on any result."
        )
        if st.button("Go to Analog Generation"):
            st.session_state["goto_page"] = "Analog Generation"
            st.rerun()
        return

    synthesis = st.session_state["synthesis_to_show"]

    try:
        with st.spinner("Rendering pathway…"):
            im = draw_synthesis(synthesis, show_intermediate=True, show_num_cases=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            im.save(tmp.name)
            st.image(tmp.name, caption="Synthesis pathway", use_container_width=True)
    except Exception as e:
        st.error(f"Could not render synthesis pathway: {e}")
        return

    # Step summary table
    pfn_list = synthesis.get_postfix_notation().to_list()
    rows = []
    for i, item in enumerate(pfn_list):
        if isinstance(item, Chem.Mol):
            smi = Chem.MolToSmiles(item, canonical=True)
            idx = item.GetProp("building_block_index") if item.HasProp("building_block_index") else "—"
            rows.append({
                "Step": i + 1, "Type": "Building Block", "Index": idx,
                "SMILES": smi, "Formula": CalcMolFormula(item),
                "MW (g/mol)": f"{Descriptors.MolWt(item):.1f}",
            })
        elif isinstance(item, Chem.rdChemReactions.ChemicalReaction):
            idx = item.GetProp("reaction_index") if item.HasProp("reaction_index") else "—"
            rows.append({
                "Step": i + 1, "Type": "Reaction", "Index": idx,
                "SMILES": "—", "Formula": "—", "MW (g/mol)": "—",
            })

    if rows:
        import pandas as pd
        st.markdown("**Step summary**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Clear synthesis view"):
        st.session_state.pop("synthesis_to_show", None)
        st.rerun()


def show_batch_processing(num_results: int, num_samples: int) -> None:
    st.subheader("Batch Processing")
    st.caption("Validate and inspect multiple molecules. Enter one SMILES per line.")

    smiles_batch = st.text_area(
        "SMILES (one per line)",
        value="CCO\nCC(=O)O\nC1CCCCC1\nC1=CC=CC=C1",
        height=140,
    )

    if st.button("Process batch", type="primary"):
        smiles_list = [s.strip() for s in smiles_batch.splitlines() if s.strip()]
        if not smiles_list:
            st.warning("Enter at least one SMILES string.")
        else:
            results = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                results.append({"smiles": smi, "mol": mol, "valid": mol is not None})
            st.session_state["batch_results"] = results

    if st.session_state.get("batch_results"):
        results = st.session_state["batch_results"]
        valid_n = sum(1 for r in results if r["valid"])
        st.info(f"{valid_n} / {len(results)} molecules are valid.")

        for i, result in enumerate(results):
            with st.expander(result["smiles"], expanded=True):
                if result["valid"]:
                    mol = result["mol"]
                    bc1, bc2 = st.columns([1, 2])
                    with bc1:
                        svg = render_mol_svg(result["smiles"])
                        if svg:
                            st.markdown(f'<div class="mol-card">{svg}</div>', unsafe_allow_html=True)
                    with bc2:
                        props = mol_properties(mol)
                        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in props.items())
                        st.markdown(f'<table class="prop-table">{rows}</table>', unsafe_allow_html=True)
                        if st.button("Generate analogs", key=f"batch_go_{i}"):
                            st.session_state["example_smiles"] = result["smiles"]
                            st.session_state["goto_page"] = "Analog Generation"
                            st.rerun()
                else:
                    st.error("Invalid SMILES.")


def show_about_page() -> None:
    st.subheader("About PrexSyn")

    st.markdown("""
    PrexSyn is a machine-learning framework for efficient, programmable exploration of
    synthesizable chemical space. Given a property query (e.g. a molecular fingerprint),
    it generates novel molecules that are guaranteed to be buildable from real building
    blocks via known reaction templates.

    **Publication**

    > Shitong Luo & Connor W. Coley.
    > *Efficient and Programmable Exploration of Synthesizable Chemical Space.*
    > arXiv 2512.00384 (2025). [arxiv.org/abs/2512.00384](https://arxiv.org/abs/2512.00384)

    **Resources**

    - [Documentation](https://prexsyn.readthedocs.io)
    - [GitHub](https://github.com/luost26/prexsyn)
    - [Training data & models](https://huggingface.co/datasets/luost26/prexsyn-data)

    **Technical stack**

    | Component | Details |
    |-----------|---------|
    | Model | Transformer-based autoregressive synthesis decoder |
    | Chemical space | Enamine building blocks + 115 reaction templates (rxn115) |
    | Fingerprints | ECFP4 (radius 2, 1024 bits) |
    | Framework | PyTorch + RDKit |
    """)

    with st.expander("BibTeX citation"):
        st.code("""\
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv:2512.00384},
  url     = {https://arxiv.org/abs/2512.00384}
}""", language="bibtex")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Auto-load model on startup
    if not st.session_state.model_loaded:
        if not load_prexsyn_model():
            st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>⚗ PrexSyn</h2>
            <p>Chemical Space Explorer</p>
        </div>
        """, unsafe_allow_html=True)

        # Programmatic navigation (e.g. from example buttons on Home)
        default_page = st.session_state.pop("goto_page", "Home")
        pages = ["Home", "Analog Generation", "Synthesis Visualization", "Batch Processing", "About"]
        default_idx = pages.index(default_page) if default_page in pages else 0

        page = st.radio("Navigate", pages, index=default_idx, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("**Generation settings**")
        num_results = st.slider("Results to show", 1, 10, 5)
        num_samples = st.slider(
            "Samples (internal)", 16, 64, 32,
            help="Larger values improve coverage but increase generation time.",
        )

        st.markdown("---")
        if st.session_state.model_loaded:
            st.markdown('<span class="status-ok">● Model ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">● Model not loaded</span>', unsafe_allow_html=True)

        if st.button("Reload model", use_container_width=True):
            st.session_state.model_loaded = False
            load_prexsyn_model()
            st.rerun()

        st.markdown("---")
        st.caption(
            "[Docs](https://prexsyn.readthedocs.io) · "
            "[Paper](https://arxiv.org/abs/2512.00384) · "
            "[GitHub](https://github.com/luost26/prexsyn)"
        )

    # Route to page
    if page == "Home":
        show_home_page()
    elif page == "Analog Generation":
        show_analog_generation(num_results, num_samples)
    elif page == "Synthesis Visualization":
        show_synthesis_visualization(num_results, num_samples)
    elif page == "Batch Processing":
        show_batch_processing(num_results, num_samples)
    elif page == "About":
        show_about_page()


if __name__ == "__main__":
    main()
