# Molecule Editor Integration Design

**Date:** 2026-02-22
**Status:** Approved
**Scope:** `prexsyn_webapp.py`, `pyproject.toml`

## Goal

Replace the non-functional SMILES-append "structure editor" (removed in the prior
rewrite) with a real, professional molecule drawing interface embedded in the
Streamlit webapp, while keeping the existing SMILES text input as an alternative.

## Chosen approach

**streamlit-ketcher** — a Streamlit component wrapping the
[Ketcher](https://lifescience.opensource.epam.com/ketcher/index.html) editor.
Ketcher is used by ChEMBL, RCSB PDB, and major cheminformatics platforms.
It provides atom-type buttons, bond-type selectors, ring templates, charge tools,
and a proper canvas — exactly the "buttons on the margins" layout requested.
`st_ketcher()` returns the drawn SMILES directly to Python; no copy-paste needed.

## UI layout — Analog Generation page

The SMILES input section becomes a two-tab widget:

```
┌─────────────────────────────────┬──────────────────────────────┐
│  Tab: Draw  │  Tab: SMILES      │                              │
├─────────────────────────────────┤   Live preview (right col)   │
│                                 │   ─────────────────────────  │
│   [  Ketcher canvas ~500px  ]   │   2D structure card          │
│                                 │   Property table             │
│   SMILES: CC(=O)Oc1ccccc1…      │   (Formula, MW, LogP, …)    │
│   (read-only badge, from draw)  │   Lipinski Ro5 pass/fail     │
│                                 │                              │
└─────────────────────────────────┴──────────────────────────────┘
│  [ Generate Analogs ]  [ Clear ]                               │
└────────────────────────────────────────────────────────────────┘
```

- **Draw tab:** `st_ketcher(value=current_smiles)` pre-populated with active SMILES.
  A small read-only code block below the canvas shows the resulting SMILES.
- **SMILES tab:** existing `st.text_input`, unchanged for power users.
- **Right column:** live SVG + property table, updates on any valid SMILES change.
- **Active SMILES:** whichever tab was last interacted with wins.

## Data flow

1. `st_ketcher()` return value → `st.session_state["draw_smiles"]`
2. Text input → `st.session_state["text_smiles"]`
3. Active tab recorded in `st.session_state["smiles_tab"]` ("draw" or "text")
4. Active SMILES = `draw_smiles` if tab == "draw", else `text_smiles`
5. Active SMILES passed to existing `process_molecule()` unchanged

## Error handling

| Condition | Behaviour |
|---|---|
| Blank canvas (empty SMILES) | Generate button disabled; right column shows placeholder |
| Ketcher returns unparseable SMILES | Inline warning; Generate button disabled |
| SMILES tab invalid text | Existing inline warning; Generate disabled |
| Switching tabs | `st_ketcher(value=...)` pre-populated so drawing is preserved |

## Dependency change

Add to `pyproject.toml` under `[tool.pixi.pypi-dependencies]`:

```toml
streamlit-ketcher = ">=0.0.1"
```

## Out of scope

- Changes to `process_molecule()`, `validate_smiles()`, `render_mol_svg()`
- Synthesis Visualization, Batch Processing, or About pages
- Any change to the core PrexSyn model or engine
