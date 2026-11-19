import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem

st.set_page_config(page_title="Solubility Predictor", page_icon="🧪")

@st.cache_resource
def load_model():
    return joblib.load("combined_rf_esol_model.joblib")

bundle = load_model()
model = bundle["model"]

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, bundle["fingerprint_radius"], nBits=bundle["fingerprint_size"]
    )
    return np.concatenate([desc, np.array(list(fp))]).reshape(1, -1)

st.title("Aqueous Solubility Predictor")
st.write("Enter a SMILES string to predict logS.")

smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O")

if st.button("Predict"):
    x = featurize(smiles.strip())
    if x is None:
        st.error("That SMILES could not be read. Please check it.")
    else:
        pred = model.predict(x)[0]
        st.metric("Predicted logS", f"{pred:.2f}")
        st.caption("Typical error on the test set: about 0.5 log units (MAE).")