#!/usr/bin/env python3
"""
PrexSyn Web Application
A comprehensive web interface for exploring chemical space with PrexSyn
"""

import streamlit as st
import pathlib
import tempfile
import torch
import re
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol

# Import PrexSyn components
from prexsyn.applications.analog import generate_analogs
from prexsyn.factories import load_model
from prexsyn.utils.draw import draw_synthesis
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.synthesis import Synthesis
from prexsyn_engine.fingerprints import tanimoto_similarity

# Page configuration
st.set_page_config(
    page_title="PrexSyn Web App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .molecule-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .similarity-badge {
        background-color: #1f77b4;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.facade = None
    st.session_state.model = None

# Load model function
def load_prexsyn_model():
    """Load the PrexSyn model"""
    try:
        model_path = pathlib.Path("./data/trained_models/v1_converted.yaml")
        if model_path.exists():
            with st.spinner("Loading PrexSyn model..."):
                facade, model = load_model(model_path, train=False)
                model = model.to("cpu")
                st.session_state.facade = facade
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ PrexSyn model loaded successfully!")
                return True
        else:
            st.error("❌ Model file not found. Please ensure trained models are available.")
            return False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return False

# Molecule visualization functions
def render_molecule(smiles, size=(300, 300)):
    """Render molecule as SVG"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
    except:
        pass
    return None

def render_3d_molecule(smiles, size=(400, 400)):
    """Render 3D molecule using py3Dmol"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            Chem.AllChem.MMFFOptimizeMolecule(mol)
            
            # Create 3D viewer
            view = py3Dmol.view(width=size[0], height=size[1])
            view.addModel(Chem.MolToMolBlock(mol), 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            return view.render()
    except:
        pass
    return None

# Main app
def main():
    """Main web application"""
    
    # Header
    st.title("🧪 PrexSyn Web Application")
    st.markdown("### Explore Chemical Space Interactively")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=PrexSyn+Web+App", width="auto")
        
        st.markdown("## 🎯 Navigation")
        page = st.radio("Select Page", ["Home", "Analog Generation", "Synthesis Visualization", "Batch Processing", "About"])
        
        st.markdown("## ⚙️ Settings")
        num_results = st.slider("Number of Results", 1, 10, 5)
        num_samples = st.slider("Number of Samples", 16, 64, 32)
        
        if st.button("🔄 Load/Reload Model"):
            st.session_state.model_loaded = False
            load_prexsyn_model()
        
        st.markdown("## 📚 Resources")
        st.markdown("""
        - [PrexSyn Documentation](https://prexsyn.readthedocs.io)
        - [PrexSyn Paper](https://arxiv.org/abs/2512.00384)
        - [GitHub Repository](https://github.com/luost26/prexsyn)
        """)
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        load_prexsyn_model()
        if not st.session_state.model_loaded:
            st.stop()
    
    # Page routing
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

def show_home_page():
    """Home page with overview"""
    st.subheader("🏠 Welcome to PrexSyn Web App")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What You Can Do
        
        - **Generate Molecular Analogs**: Find similar molecules to any input structure
        - **Visualize Synthesis Pathways**: See how molecules can be synthesized step-by-step
        - **Explore Chemical Space**: Navigate through synthesizable chemical space
        - **Batch Processing**: Process multiple molecules at once
        - **3D Visualization**: View molecules in interactive 3D
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Features
        
        - **Interactive Interface**: User-friendly web interface
        - **Real-time Results**: Instant feedback and visualization
        - **Multiple Visualizations**: 2D structures, 3D models, synthesis diagrams
        - **Customizable Parameters**: Adjust settings for your needs
        - **No Coding Required**: Access all functionality through the web UI
        """)
    
    st.markdown("### 🎓 Quick Start")
    st.markdown("""
    1. **Navigate** to "Analog Generation" in the sidebar
    2. **Enter** a SMILES string (e.g., "CC(=O)OC1=CC=CC=C1C(=O)O" for aspirin)
    3. **Click** "Generate Analogs"
    4. **Explore** the results with interactive visualizations
    """)
    
    # Example molecules
    st.markdown("### 🧪 Try These Example Molecules")
    
    examples = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Benzene": "C1=CC=CC=C1",
        "Ethanol": "CCO"
    }
    
    cols = st.columns(3)
    for i, (name, smiles) in enumerate(examples.items()):
        with cols[i % 3]:
            if st.button(f"🔬 {name}", key=f"example_{i}"):
                st.session_state['example_smiles'] = smiles
                st.session_state['current_page'] = 'Analog Generation'
                st.rerun()

def show_analog_generation(num_results, num_samples):
    """Analog generation page"""
    st.header("🔬 Molecular Analog Generation")
    
    st.markdown("""
    Generate synthesizable analogs of any molecule. Enter a SMILES string below and explore 
    similar molecules that can be synthesized.
    """)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES string:",
            value=st.session_state.get('example_smiles', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
            help="Example: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)"
        )
    
    with col2:
        if st.button("🎲 Random Molecule"):
            # Generate a random simple molecule
            import random
            random_smiles = random.choice([
                "CCO", "CC(=O)O", "C1CCCCC1", "C1=CC=CC=C1", 
                "C1=CC=NC=N1", "CC(C)O", "COC", "CN"
            ])
            smiles_input = random_smiles
            st.rerun()
    
    # Generate button
    if st.button("🚀 Generate Analogs", type="primary"):
        if smiles_input:
            with st.spinner("Generating analogs..."):
                process_molecule(smiles_input, num_results, num_samples)
        else:
            st.warning("Please enter a SMILES string")
    
    # Clear button
    if st.button("🧹 Clear Results"):
        st.session_state['results'] = None
        st.rerun()
    
    # Display results if available
    if 'results' in st.session_state and st.session_state['results']:
        display_results(st.session_state['results'], num_results)

def validate_smiles(smiles):
    """Validate SMILES string format"""
    # Basic SMILES pattern validation
    if not smiles or len(smiles) > 2000:  # Prevent excessively long inputs
        return False
    
    # Check for potentially dangerous characters
    dangerous_patterns = [";", "&", "|", "<", ">", "$", "`", "'", '"', "\\", "\n", "\r"]
    if any(pattern in smiles for pattern in dangerous_patterns):
        return False
    
    # Basic SMILES character validation
    valid_chars = r'^[A-Za-z0-9@+\-=\[\]\(\)\/%=#$\.]+$'
    return bool(re.match(valid_chars, smiles))

def process_molecule(smiles, num_results, num_samples):
    """Process a molecule and generate analogs"""
    try:
        # Validate SMILES input
        if not validate_smiles(smiles):
            st.error("❌ Invalid SMILES format or potentially dangerous input")
            return
        
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error(f"❌ Invalid SMILES: {smiles}")
            return
        
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        
        # Create sampler
        sampler = BasicSampler(
            st.session_state.model,
            token_def=st.session_state.facade.tokenization.token_def,
            num_samples=num_samples,
            max_length=16,
        )
        
        # Generate analogs
        result = generate_analogs(
            facade=st.session_state.facade,
            model=st.session_state.model,
            sampler=sampler,
            fp_property=st.session_state.facade.property_set["ecfp4"],
            mol=mol,
        )
        
        # Process results
        visited = set()
        result_list = []
        for synthesis in result["synthesis"]:
            if synthesis.stack_size() != 1:
                continue
            for prod in synthesis.top().to_list():
                prod_smi = Chem.MolToSmiles(prod, canonical=True)
                if prod_smi in visited:
                    continue
                visited.add(prod_smi)
                sim = tanimoto_similarity(prod, mol, fp_type="ecfp4")
                result_list.append((prod, synthesis, sim))
        
        # Sort by similarity
        result_list.sort(key=lambda x: x[2], reverse=True)
        
        # Store results
        st.session_state['results'] = {
            'input_smiles': canonical_smi,
            'input_mol': mol,
            'results': result_list[:num_results]
        }
        
    except Exception as e:
        st.error(f"❌ Error generating analogs: {e}")

def display_results(results_data, num_results):
    """Display the results in a nice format"""
    
    # Input molecule
    st.markdown("### 🎯 Input Molecule")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        svg = render_molecule(results_data['input_smiles'])
        if svg:
            st.markdown(f"<div class='molecule-container'>{svg}</div>", unsafe_allow_html=True)
        else:
            st.code(results_data['input_smiles'])
    
    with col2:
        st.markdown(f"**SMILES:** `{results_data['input_smiles']}`")
        st.markdown(f"**Formula:** {Chem.CalcMolFormula(results_data['input_mol'])}")
        st.markdown(f"**MW:** {Chem.Descriptors.MolWt(results_data['input_mol']):.2f} g/mol")
    
    # Results
    st.markdown(f"### 🔬 Top {len(results_data['results'])} Analogs")
    
    for i, (prod, synthesis, sim) in enumerate(results_data['results']):
        with st.expander(f"Result {i+1}: Similarity {sim:.3f}", expanded=True):
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                svg = render_molecule(prod_smi)
                if svg:
                    st.markdown(f"<div class='molecule-container'>{svg}</div>", unsafe_allow_html=True)
                else:
                    st.code(prod_smi)
                
                # 3D visualization
                if st.checkbox(f"Show 3D View", key=f"3d_{i}"):
                    html_3d = render_3d_molecule(prod_smi)
                    if html_3d:
                        st.components.v1.html(html_3d, height=400)
            
            with col2:
                st.markdown(f"**SMILES:** `{prod_smi}`")
                st.markdown(f"**Formula:** {Chem.CalcMolFormula(prod)}")
                st.markdown(f"**MW:** {Chem.Descriptors.MolWt(prod):.2f} g/mol")
                st.markdown(f"**Similarity:** <span class='similarity-badge'>{sim:.3f}</span>", unsafe_allow_html=True)
                
                if st.button(f"🔍 View Synthesis Pathway", key=f"synth_{i}"):
                    st.session_state['synthesis_to_show'] = synthesis
                    st.session_state['current_page'] = 'Synthesis Visualization'
                    st.rerun()

def show_synthesis_visualization(num_results, num_samples):
    """Synthesis visualization page"""
    st.header("🧬 Synthesis Pathway Visualization")
    
    st.markdown("""
    Visualize the synthesis pathways for generated molecules. This shows how each molecule 
    can be constructed from building blocks using chemical reactions.
    """)
    
    # Check if we have a synthesis to show
    if 'synthesis_to_show' in st.session_state and st.session_state['synthesis_to_show']:
        synthesis = st.session_state['synthesis_to_show']
        
        try:
            # Generate synthesis image
            with st.spinner("Generating synthesis pathway..."):
                im = draw_synthesis(synthesis, show_intermediate=True, show_num_cases=True)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    im.save(tmp.name)
                    st.image(tmp.name, caption="Synthesis Pathway", width="auto")
                
                # Synthesis details
                st.markdown("### 📋 Synthesis Details")
                
                # Get synthesis information
                replay = Synthesis()
                pfn_list = synthesis.get_postfix_notation().to_list()
                
                synthesis_text = """
                **Synthesis Steps:**
                """
                
                for i, item in enumerate(pfn_list):
                    if isinstance(item, Chem.Mol):
                        smi = Chem.MolToSmiles(item, canonical=True)
                        idx = item.GetProp("building_block_index")
                        synthesis_text += f"\n{i+1}. Building Block: {smi} (Index: {idx})"
                        if item.HasProp("id"):
                            synthesis_text += f" (ID: {item.GetProp('id')})"
                    elif isinstance(item, Chem.rdChemReactions.ChemicalReaction):
                        idx = item.GetProp("reaction_index")
                        synthesis_text += f"\n{i+1}. Reaction: Index {idx}"
                
                st.text_area("Synthesis Pathway Details", synthesis_text, height=200)
                
        except Exception as e:
            st.error(f"❌ Error generating synthesis visualization: {e}")
            st.code(f"Synthesis object: {synthesis}")
    else:
        st.info("📋 No synthesis pathway selected. Generate analogs first and click 'View Synthesis Pathway' on a result.")
        
        if st.button("🔙 Go to Analog Generation"):
            st.session_state['current_page'] = 'Analog Generation'
            st.rerun()

def show_batch_processing(num_results, num_samples):
    """Batch processing page"""
    st.header("🔄 Batch Processing")
    
    st.markdown("""
    Process multiple molecules at once. Enter SMILES strings separated by lines.
    """)
    
    # Text area for multiple SMILES
    smiles_batch = st.text_area(
        "Enter SMILES strings (one per line):",
        value="CCO\nCC(=O)O\nC1CCCCC1\nC1=CC=CC=C1",
        height=150
    )
    
    if st.button("🚀 Process Batch", type="primary"):
        if smiles_batch.strip():
            smiles_list = [s.strip() for s in smiles_batch.split('\n') if s.strip()]
            
            # Validate all SMILES before processing
            invalid_smiles = []
            for smiles in smiles_list:
                if not validate_smiles(smiles):
                    invalid_smiles.append(smiles)
            
            if invalid_smiles:
                st.error(f"❌ Invalid SMILES detected: {', '.join(invalid_smiles)}")
                st.stop()
            
            with st.spinner(f"Processing {len(smiles_list)} molecules..."):
                batch_results = []
                
                for i, smiles in enumerate(smiles_list):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            batch_results.append({
                                'smiles': smiles,
                                'mol': mol,
                                'valid': True
                            })
                        else:
                            batch_results.append({
                                'smiles': smiles,
                                'valid': False,
                                'error': 'Invalid SMILES'
                            })
                    except Exception as e:
                        batch_results.append({
                            'smiles': smiles,
                            'valid': False,
                            'error': str(e)
                        })
                
                st.session_state['batch_results'] = batch_results
                st.success(f"✅ Processed {len(batch_results)} molecules")
        else:
            st.warning("Please enter at least one SMILES string")
    
    # Display batch results
    if 'batch_results' in st.session_state and st.session_state['batch_results']:
        st.markdown(f"### Results ({len(st.session_state['batch_results'])} molecules)")
        
        for i, result in enumerate(st.session_state['batch_results']):
            with st.expander(f"Molecule {i+1}: {result['smiles']}", expanded=True):
                if result['valid']:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        svg = render_molecule(result['smiles'])
                        if svg:
                            st.markdown(f"<div class='molecule-container'>{svg}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**SMILES:** `{result['smiles']}`")
                        st.markdown(f"**Formula:** {Chem.CalcMolFormula(result['mol'])}")
                        st.markdown(f"**MW:** {Chem.Descriptors.MolWt(result['mol']):.2f} g/mol")
                        
                        if st.button(f"🔬 Generate Analogs", key=f"batch_analog_{i}"):
                            st.session_state['example_smiles'] = result['smiles']
                            st.session_state['current_page'] = 'Analog Generation'
                            st.rerun()
                else:
                    st.error(f"❌ Invalid molecule: {result.get('error', 'Unknown error')}")

def show_about_page():
    """About page"""
    st.header("📚 About PrexSyn Web App")
    
    st.markdown("""
    ## Welcome to the PrexSyn Web Application!
    
    This web interface provides easy access to PrexSyn's powerful chemical space exploration 
    capabilities without requiring any coding knowledge.
    
    ### 🎯 Key Features
    
    - **Interactive Interface**: User-friendly web interface for all PrexSyn functionality
    - **Molecular Visualization**: 2D and 3D visualization of molecules
    - **Analog Generation**: Find synthesizable analogs of any molecule
    - **Synthesis Pathways**: Visualize how molecules can be synthesized
    - **Batch Processing**: Process multiple molecules simultaneously
    - **Real-time Results**: Instant feedback and visualization
    
    ### 🔬 About PrexSyn
    
    PrexSyn is an efficient, accurate, and programmable framework for exploring synthesizable 
    chemical space. It uses advanced machine learning and cheminformatics techniques to:
    
    - Generate novel molecules that can actually be synthesized
    - Plan synthesis pathways using available building blocks
    - Optimize molecules for desired properties
    - Explore vast chemical spaces efficiently
    
    ### 📖 Resources
    
    - **Documentation**: [https://prexsyn.readthedocs.io](https://prexsyn.readthedocs.io)
    - **Paper**: [https://arxiv.org/abs/2512.00384](https://arxiv.org/abs/2512.00384)
    - **GitHub**: [https://github.com/luost26/prexsyn](https://github.com/luost26/prexsyn)
    - **Data**: [https://huggingface.co/datasets/luost26/prexsyn-data](https://huggingface.co/datasets/luost26/prexsyn-data)
    
    ### 🛠️ Technical Details
    
    - **Framework**: Streamlit web application
    - **Backend**: PrexSyn with PyTorch and RDKit
    - **Visualization**: RDKit 2D rendering, py3Dmol for 3D
    - **Chemical Space**: Enamine building blocks and reactions
    
    ### 🤝 Citation
    
    If you use PrexSyn in your research, please cite:
    
    ```bibtex
    @article{luo2025prexsyn,
      title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
      author  = {Shitong Luo and Connor W. Coley},
      year    = {2025},
      journal = {arXiv preprint arXiv: 2512.00384}
    }
    ```
    
    ### 📧 Contact
    
    For questions or support, please refer to the GitHub repository or the documentation.
    """)

# Run the app
if __name__ == "__main__":
    main()