# Molecule Editor (Ketcher) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the plain SMILES text input on the Analog Generation page with a
two-tab Draw / SMILES widget that embeds the Ketcher molecule editor, keeping the
existing SMILES text input as an alternative, with a live property preview in a
right-hand column that reacts to whichever input is active.

**Architecture:** `streamlit-ketcher` wraps Ketcher as a proper Streamlit component;
`st_ketcher()` renders the canvas and returns the current SMILES on every re-run.
Active-input tracking is done entirely through `st.session_state` (no callbacks
needed for the draw tab; `on_change` on the text input marks the text tab active).
The form wrapper on `show_analog_generation()` is removed because interactive
components cannot be nested inside `st.form`.

**Tech Stack:** `streamlit-ketcher` (PyPI), RDKit, Streamlit ≥ 1.30, pixi

---

### Task 1: Add the dependency

**Files:**
- Modify: `pyproject.toml` (lines 78-79, `[tool.pixi.pypi-dependencies]` section)

**Step 1: Add `streamlit-ketcher` to pyproject.toml**

In `pyproject.toml`, find the `[tool.pixi.pypi-dependencies]` table (currently has
only `prexsyn`) and add the new entry so the block reads:

```toml
[tool.pixi.pypi-dependencies]
prexsyn = { path = ".", editable = true }
streamlit-ketcher = ">=0.0.1"
```

**Step 2: Install**

```bash
pixi install
```

Expected: pixi resolves and installs `streamlit-ketcher` (and its transitive deps)
without conflicts. The output ends with something like `✔ All packages installed`.

**Step 3: Smoke-test the import**

```bash
pixi run python -c "from streamlit_ketcher import st_ketcher; print('OK')"
```

Expected output: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "feat: add streamlit-ketcher dependency"
```

---

### Task 2: Rewrite `show_analog_generation()`

**Files:**
- Modify: `prexsyn_webapp.py` — replace the entire `show_analog_generation()` function
  (currently lines 322-382) and add the `st_ketcher` import at the top of the file.

**Step 1: Add the import**

At the top of `prexsyn_webapp.py`, after the existing imports, add:

```python
from streamlit_ketcher import st_ketcher
```

**Step 2: Replace `show_analog_generation()`**

Delete the current `show_analog_generation()` function (lines 322-382) and replace
it with the implementation below.

Key design notes before reading the code:
- `st.form()` is **removed** — interactive components like `st_ketcher` cannot live
  inside a Streamlit form; buttons are standalone instead.
- Active-input tracking: the text `st.text_input` uses `on_change` to write
  `"text"` into `st.session_state["active_input"]`. For the draw tab, we compare
  `st_ketcher()`'s return value against the previously stored drawn SMILES; if they
  differ the user has just drawn something, so we update state and mark draw active.
- `st_ketcher()` is pre-populated via `molecule=` so switching tabs doesn't lose work.
- The Generate button is **disabled** (not hidden) when there is no valid SMILES, so
  the user gets clear feedback rather than a silent no-op.

```python
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
            drawn = st_ketcher(molecule=seed_smiles, height=500, key="ketcher")

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
```

**Step 3: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('prexsyn_webapp.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

**Step 4: Commit**

```bash
git add prexsyn_webapp.py
git commit -m "feat: add Ketcher molecule editor to Analog Generation page

Two-tab Draw / SMILES input with live property preview.
Removes st.form wrapper (incompatible with interactive components).
Active-input tracking via session_state; Generate button disabled
when no valid molecule is present."
```

---

### Task 3: Manual verification

Start the app:

```bash
pixi run streamlit run prexsyn_webapp.py
```

Work through this checklist in the browser:

| Check | Expected |
|---|---|
| Page loads without errors | Analog Generation shows two tabs |
| Draw tab shows Ketcher canvas | Full editor with atom/bond toolbar visible |
| Draw benzene ring in Ketcher | Right column updates: 2D SVG + MW ~78, formula C6H6 |
| SMILES code block below canvas | Shows `c1ccccc1` or equivalent |
| Switch to SMILES tab, type `CCO` | Right column updates: ethanol, MW ~46 |
| Switch back to Draw tab | Canvas still shows previous drawing (not reset) |
| Click Generate Analogs (Draw active) | Spinner runs, results appear |
| Click Clear | Both tabs reset, results gone, Generate disabled |
| Home page example button (Aspirin) | Navigates to Analog Generation, Ketcher pre-loaded with aspirin SMILES |
| Blank canvas (erase all) | Generate button disabled, right column shows info placeholder |

**No automated UI tests are needed** — Streamlit components cannot be unit-tested
headlessly; the above manual checklist covers all acceptance criteria.

---

### Task 4: Update memory

After verifying everything works, update
`/home/christopher/.claude/projects/-home-christopher-git-prexsyn/memory/MEMORY.md`
to record:

- `streamlit-ketcher` is now a PyPI dependency under `[tool.pixi.pypi-dependencies]`
- `show_analog_generation()` uses two tabs: Draw (Ketcher) + SMILES text
- Active-input tracking via `st.session_state["active_input"]` ("draw" or "text")
- `st.form` is intentionally absent from this page (incompatible with `st_ketcher`)
