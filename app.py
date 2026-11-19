from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Draw, rdFingerprintGenerator, rdMolDescriptors

st.set_page_config(page_title="Esol: Solubility Predictor", page_icon="🧪", layout="wide")

BASE = Path(__file__).parent
MODEL_FILE = BASE / "combined_rf_esol_model.joblib"
TEST_FILE = BASE / "combined_rf_test_predictions.csv"
METRICS_FILE = BASE / "model_metrics.csv"

ALLOWED = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Anthracene": "c1ccc2cc3ccccc3cc2c1",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
}

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1.1rem 1.3rem;
       background: rgba(128,128,128,.06);}
.label {font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; opacity: .65;}
.big {font-size: 2.8rem; font-weight: 700; line-height: 1.1; margin: .2rem 0;}
.scale {height: 10px; border-radius: 6px; position: relative; margin: 1.3rem 0 .4rem;
        background: linear-gradient(90deg,#c0392b,#e67e22,#f1c40f,#2e9e6b,#1f78b4);}
.marker {position: absolute; top: -6px; width: 5px; height: 22px; background: #222;
         border: 2px solid #fff; border-radius: 3px;}
.ends {display: flex; justify-content: space-between; font-size: .75rem; opacity: .65;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- loading ----------
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_test():
    return pd.read_csv(TEST_FILE) if TEST_FILE.exists() else None


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_FILE) if METRICS_FILE.exists() else None


bundle = load_bundle()
model = bundle["model"]
DESC_COLS = bundle["descriptor_columns"]
RADIUS = bundle["fingerprint_radius"]
NBITS = bundle["fingerprint_size"]
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
test_df = load_test()
metrics_df = load_metrics()


@st.cache_resource
def test_fingerprints():
    if test_df is None:
        return []
    return [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in test_df["smiles"]]


# ---------- chemistry helpers ----------
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    desc = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    x = np.concatenate([[desc[c] for c in DESC_COLS], fpgen.GetFingerprintAsNumPy(mol)])
    return {"mol": mol, "desc": desc, "x": x.astype(float), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)}


def predict_matrix(X):
    trees = np.stack([t.predict(X) for t in model.estimators_])
    return trees.mean(axis=0), trees.std(axis=0)


def solubility_class(logs):
    if logs >= -1:
        return "Highly soluble", "Roughly 0.1 mol/L or more dissolves. Dissolving this compound in water should be easy."
    if logs >= -3:
        return "Soluble", "It should dissolve at normal lab concentrations, from about 1 mmol/L up to 0.1 mol/L."
    if logs >= -5:
        return "Poorly soluble", "Only small amounts dissolve. In drug work this range often needs formulation help."
    return "Very poorly soluble", "Almost nothing dissolves in water. Expect it to stay as a solid or oil."


def confidence_text(std):
    if std < 0.6:
        return "High agreement", "The trees in the forest give very similar answers, so the model is fairly settled on this one."
    if std < 0.9:
        return "Moderate agreement", "The trees differ a little. Treat the number as a good estimate, not an exact value."
    return "Low agreement", "The trees disagree a lot. This molecule may look unlike what the model learned, so be careful with the result."


def input_flags(mol, smiles):
    out = []
    if "." in smiles:
        out.append("The input has more than one component, such as a salt. The model was built for single neutral molecules.")
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        out.append("The molecule carries a charge. The model mostly saw neutral compounds.")
    odd = {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED
    if odd:
        out.append("It contains uncommon elements (" + ", ".join(sorted(odd)) + "). Predictions here are unreliable.")
    mw = Descriptors.MolWt(mol)
    if mw > 700 or mw < 30:
        out.append(f"Molecular weight is {mw:.0f}, which is outside the range of typical training molecules.")
    return out


def explain_factors(d, n_arom):
    pts = []
    lp = d["MolLogP"]
    if lp >= 3:
        pts.append(("down", "Lipophilicity", f"LogP is {lp:.1f}. The molecule is oily and prefers fat-like surroundings to water, which usually lowers solubility."))
    elif lp <= 1:
        pts.append(("up", "Lipophilicity", f"LogP is {lp:.1f}. The molecule is water-friendly, which usually helps it dissolve."))
    else:
        pts.append(("flat", "Lipophilicity", f"LogP is {lp:.1f}, a middle value that does not push strongly either way."))
    tp = d["TPSA"]
    if tp >= 90:
        pts.append(("up", "Polar surface", f"TPSA is {tp:.0f}. A large polar surface gives water plenty to bind to."))
    elif tp <= 30:
        pts.append(("down", "Polar surface", f"TPSA is {tp:.0f}. There is little polar surface for water to interact with."))
    else:
        pts.append(("flat", "Polar surface", f"TPSA is {tp:.0f}, a moderate amount of polar surface."))
    mw = d["MolWt"]
    if mw >= 400:
        pts.append(("down", "Size", f"Molecular weight is {mw:.0f}. Larger molecules are harder to pull into solution."))
    elif mw <= 200:
        pts.append(("up", "Size", f"Molecular weight is {mw:.0f}. Small molecules are easier for water to surround."))
    else:
        pts.append(("flat", "Size", f"Molecular weight is {mw:.0f}, a mid-sized molecule."))
    hb = d["HBD"] + d["HBA"]
    if hb >= 6:
        pts.append(("up", "Hydrogen bonding", f"It has {d['HBD']} donor and {d['HBA']} acceptor sites, so it can bond with water in many places."))
    elif hb <= 1:
        pts.append(("down", "Hydrogen bonding", f"It has {d['HBD']} donor and {d['HBA']} acceptor sites, so there is little for water to grab onto."))
    else:
        pts.append(("flat", "Hydrogen bonding", f"It has {d['HBD']} donor and {d['HBA']} acceptor sites."))
    if n_arom >= 2:
        pts.append(("down", "Aromatic rings", f"There are {n_arom} aromatic rings. Flat ring systems stack tightly in the solid, which makes them harder to dissolve."))
    elif n_arom == 0 and d["FractionCSP3"] >= 0.5:
        pts.append(("up", "Shape", "The structure is mostly saturated and non-flat, so it packs less tightly as a solid."))
    return pts


def solubility_units(logs, mw):
    mol_l = 10 ** logs
    return mol_l, mol_l * mw  # mol/L and g/L (same as mg/mL)


def fmt_sci(v):
    return f"{v:.2e}" if (v < 0.001 or v >= 1000) else f"{v:.3g}"


# ---------- layout ----------
st.title("Esol: Aqueous Solubility Predictor")
st.write(
    "Type in a molecule as a SMILES string and the model estimates how well it dissolves in water. "
    "The result is shown as logS, the base-10 logarithm of solubility in mol/L, and explained in plain terms."
)

with st.sidebar:
    st.header("About the model")
    st.write("Tuned Random Forest trained on 8 molecular descriptors plus a 2048-bit Morgan fingerprint (radius 2).")
    if metrics_df is not None:
        best = metrics_df.iloc[0]
        st.metric("R² on test set", f"{best['R2']:.2f}")
        st.metric("Mean absolute error", f"{best['MAE']:.2f} log units")
    st.caption("Research tool. Do not use these numbers as a replacement for lab measurements.")
    st.divider()
    st.caption("Built by Agboola Abdulroheem Oyindamola, Computational Chemist")

tab_predict, tab_batch, tab_perf, tab_about, tab_dev = st.tabs(["Predict", "Batch", "Model performance", "How it works", "About the developer"])

# ---------- tab 1: predict ----------
with tab_predict:
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = EXAMPLES["Aspirin"]

    def use_example():
        choice = st.session_state.example_choice
        if choice in EXAMPLES:
            st.session_state.smiles_input = EXAMPLES[choice]

    c1, c2 = st.columns([3, 1.3])
    c1.text_input("SMILES string", key="smiles_input", help="Press Enter to run the prediction.")
    c2.selectbox("Or pick an example", ["Choose an example"] + list(EXAMPLES), key="example_choice", on_change=use_example)

    smiles = st.session_state.smiles_input.strip()
    res = featurize(smiles)

    if not smiles:
        st.info("Enter a SMILES string to begin.")
    elif res is None:
        st.error("That SMILES could not be read. Check for typos, unclosed rings or brackets.")
    else:
        X = res["x"].reshape(1, -1)
        mean, std = predict_matrix(X)
        pred, spread = float(mean[0]), float(std[0])
        mol_l, g_l = solubility_units(pred, res["desc"]["MolWt"])
        cls, cls_text = solubility_class(pred)
        conf, conf_text = confidence_text(spread)

        for f in input_flags(res["mol"], smiles):
            st.warning(f)

        left, right = st.columns([1, 1.25])
        with left:
            st.image(Draw.MolToImage(res["mol"], size=(420, 300)))
        with right:
            pos = (min(max(pred, -8), 1) + 8) / 9 * 100
            st.markdown(
                f"""
<div class="card">
<div class="label">Predicted logS</div>
<div class="big">{pred:.2f}</div>
<div><b>{cls}</b></div>
<div class="scale"><div class="marker" style="left:calc({pos:.1f}% - 3px)"></div></div>
<div class="ends"><span>-8 (insoluble)</span><span>1 (very soluble)</span></div>
</div>
""",
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Solubility (mol/L)", fmt_sci(mol_l))
        m2.metric("Solubility (mg/mL)", fmt_sci(g_l))
        m3.metric("Typical range", f"{pred - spread:.2f} to {pred + spread:.2f}", help="Prediction plus or minus one standard deviation across the trees in the forest. On the test set the measured value fell inside this range for about 78% of molecules.")

        st.subheader("What this result means")
        st.write(cls_text)
        st.write(f"**Model confidence: {conf}.** {conf_text}")
        if test_df is not None:
            within = (test_df["residual"].abs() <= 0.5).mean() * 100
            within1 = (test_df["residual"].abs() <= 1.0).mean() * 100
            st.write(
                f"On the held-out test set, {within:.0f}% of predictions landed within 0.5 log units of the measured value "
                f"and {within1:.0f}% within 1.0 log unit. A log unit is a tenfold change in solubility, so a difference of 0.5 means about a threefold gap."
            )

        st.subheader("Why the molecule behaves this way")
        st.caption("These are general chemistry trends read from the molecule's properties. They are not the model's internal reasoning.")
        for direction, title, text in explain_factors(res["desc"], res["aromatic_rings"]):
            tag = {"up": ":green[Helps solubility]", "down": ":red[Hurts solubility]", "flat": ":gray[Neutral]"}[direction]
            st.markdown(f"**{title}** {tag}  \n{text}")

        with st.expander("Calculated descriptors"):
            dd = pd.DataFrame({"Descriptor": list(res["desc"].keys()), "Value": [round(v, 3) for v in res["desc"].values()]})
            st.dataframe(dd, hide_index=True)

        if test_df is not None:
            st.subheader("Similar molecules with known solubility")
            sims = DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(res["mol"]), test_fingerprints())
            top = np.argsort(sims)[::-1][:5]
            near = test_df.iloc[top][["smiles", "logS", "predicted_logS"]].copy()
            near.insert(1, "Similarity", [round(sims[i], 2) for i in top])
            near.columns = ["SMILES", "Similarity", "Measured logS", "Model's logS"]
            st.dataframe(near, hide_index=True)
            st.caption("These come from the held-out test set. Similarity is the Tanimoto score, where 1.0 is identical. Low scores mean no close relatives were found.")

            fig = go.Figure()
            fig.add_histogram(x=test_df["logS"], nbinsx=30, marker_color="rgba(31,120,180,.55)", name="Test molecules")
            fig.add_vline(x=pred, line_width=3, line_color="#c0392b", annotation_text="Your molecule", annotation_position="top")
            fig.update_layout(title="Where your molecule sits among test compounds", xaxis_title="logS", yaxis_title="Count", height=320, margin=dict(t=50, b=40), showlegend=False)
            st.plotly_chart(fig)

# ---------- tab 2: batch ----------
with tab_batch:
    st.write("Predict many molecules at once. Upload a CSV with a column named `smiles`, or paste one SMILES per line.")
    up = st.file_uploader("CSV file", type="csv")
    pasted = st.text_area("Or paste SMILES here", height=120, placeholder="CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O")

    smiles_list = []
    if up is not None:
        raw = pd.read_csv(up)
        col = next((c for c in raw.columns if c.strip().lower() == "smiles"), None)
        if col is None:
            st.error("No column named 'smiles' was found in that file.")
        else:
            smiles_list = raw[col].astype(str).str.strip().tolist()
    elif pasted.strip():
        smiles_list = [s.strip() for s in pasted.splitlines() if s.strip()]

    if smiles_list:
        if len(smiles_list) > 2000:
            st.warning("Only the first 2000 molecules will be used.")
            smiles_list = smiles_list[:2000]
        feats = [featurize(s) for s in smiles_list]
        ok = [i for i, f in enumerate(feats) if f is not None]
        rows = [{"smiles": s, "predicted_logS": None, "mg_per_mL": None, "class": "Invalid SMILES", "tree_spread": None} for s in smiles_list]
        if ok:
            m, s_ = predict_matrix(np.stack([feats[i]["x"] for i in ok]))
            for k, i in enumerate(ok):
                p = float(m[k])
                rows[i].update(
                    predicted_logS=round(p, 3),
                    mg_per_mL=float(f"{solubility_units(p, feats[i]['desc']['MolWt'])[1]:.3g}"),
                    **{"class": solubility_class(p)[0]},
                    tree_spread=round(float(s_[k]), 3),
                )
        out = pd.DataFrame(rows)
        st.dataframe(out, hide_index=True)
        bad = len(smiles_list) - len(ok)
        st.caption(f"{len(ok)} predicted, {bad} could not be read.")
        st.download_button("Download results as CSV", out.to_csv(index=False).encode(), "esol_predictions.csv", "text/csv")
        if ok:
            fig = go.Figure(go.Histogram(x=out["predicted_logS"].dropna(), nbinsx=25, marker_color="rgba(31,120,180,.6)"))
            fig.update_layout(title="Distribution of predicted logS", xaxis_title="Predicted logS", yaxis_title="Count", height=300, margin=dict(t=50, b=40))
            st.plotly_chart(fig)

# ---------- tab 3: performance ----------
with tab_perf:
    if metrics_df is None or test_df is None:
        st.info("Add model_metrics.csv and combined_rf_test_predictions.csv to the app folder to see this tab.")
    else:
        best = metrics_df.iloc[0]
        st.subheader("How well does it work?")
        a, b, c = st.columns(3)
        a.metric("R²", f"{best['R2']:.3f}")
        b.metric("MAE", f"{best['MAE']:.3f}")
        c.metric("RMSE", f"{best['RMSE']:.3f}")
        with st.expander("What do these numbers mean?"):
            st.write(
                f"**R²** tells you how much of the variation in solubility the model captures. {best['R2']:.2f} means about {best['R2']*100:.0f}% of it, "
                "where 1.0 would be perfect and 0 is no better than guessing the average."
            )
            st.write(
                f"**MAE** (mean absolute error) is the average size of the miss, {best['MAE']:.2f} log units here. "
                f"**RMSE** is similar but punishes large misses more, and at {best['RMSE']:.2f} it is higher because a few molecules are badly off."
            )

        st.subheader("Measured vs predicted")
        lo, hi = float(min(test_df["logS"].min(), test_df["predicted_logS"].min())), float(max(test_df["logS"].max(), test_df["predicted_logS"].max()))
        fig = go.Figure()
        fig.add_scatter(x=test_df["logS"], y=test_df["predicted_logS"], mode="markers", text=test_df["smiles"],
                        marker=dict(size=7, color="rgba(31,120,180,.65)"), name="Test molecules",
                        hovertemplate="%{text}<br>Measured %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>")
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#c0392b", dash="dash"), name="Perfect prediction")
        fig.update_layout(xaxis_title="Measured logS", yaxis_title="Predicted logS", height=480, margin=dict(t=30))
        st.plotly_chart(fig)
        st.caption("Points on the dashed line are perfect. The model tracks the trend well. It tends to be least accurate for the most soluble compounds (logS above 0), where a few points sit far below the line.")

        l, r = st.columns(2)
        with l:
            fig = go.Figure(go.Histogram(x=test_df["residual"], nbinsx=30, marker_color="rgba(31,120,180,.6)"))
            fig.update_layout(title="Errors (measured minus predicted)", xaxis_title="Residual", height=320, margin=dict(t=50))
            st.plotly_chart(fig)
            st.caption("A bell shape centred near zero means the model is not biased high or low overall.")
        with r:
            fig = go.Figure(go.Bar(x=metrics_df["Model"], y=metrics_df["MAE"], marker_color="rgba(31,120,180,.7)"))
            fig.update_layout(title="Error by model (lower is better)", yaxis_title="MAE", height=320, margin=dict(t=50))
            st.plotly_chart(fig)
            st.caption("The tuned Random Forest beat the stacked model, the descriptor-only forest, a Ridge model and the plain average.")

        st.subheader("What the model relies on")
        imp = model.feature_importances_
        d_imp = pd.Series(imp[: len(DESC_COLS)], index=DESC_COLS).sort_values()
        fp_total = float(imp[len(DESC_COLS):].sum())
        fig = go.Figure(go.Bar(x=d_imp.values, y=d_imp.index, orientation="h", marker_color="rgba(31,120,180,.7)"))
        fig.update_layout(title="Importance of each descriptor", height=340, margin=dict(t=50))
        st.plotly_chart(fig)
        st.caption(f"All 2048 fingerprint bits together account for {fp_total*100:.0f}% of the total importance. Importance shows how much the forest used a feature, not whether it raises or lowers solubility.")

        with st.expander("Full comparison table"):
            st.dataframe(metrics_df.round(3), hide_index=True)

# ---------- tab 4: about ----------
with tab_about:
    st.subheader("From structure to prediction")
    st.markdown(
        """
1. **SMILES in.** A SMILES string is a line of text that describes a molecule's structure, such as `CCO` for ethanol.
2. **Features out.** RDKit reads the structure and calculates eight descriptors: molecular weight, LogP, TPSA, hydrogen bond donors and acceptors, rotatable bonds, ring count and the fraction of sp3 carbons. It also builds a 2048-bit Morgan fingerprint that records which small substructures are present.
3. **Forest votes.** A Random Forest is a large group of decision trees. Each tree makes its own prediction and the average becomes the answer. The spread between trees is the confidence indicator.
4. **Translate.** The logS value is converted to mol/L and mg/mL and sorted into a plain-language class.
"""
    )
    st.subheader("Reading logS")
    st.write(
        "logS is the base-10 logarithm of solubility in mol/L. A logS of -2 means 0.01 mol/L, and -4 means 0.0001 mol/L. "
        "Each step down is ten times less soluble."
    )
    st.table(pd.DataFrame({
        "logS": ["-1 or higher", "-1 to -3", "-3 to -5", "Below -5"],
        "Class": ["Highly soluble", "Soluble", "Poorly soluble", "Very poorly soluble"],
        "Rough meaning": ["0.1 mol/L or more", "About 1 mmol/L to 0.1 mol/L", "About 10 µmol/L to 1 mmol/L", "Below about 10 µmol/L"],
    }))
    st.subheader("Limits to keep in mind")
    st.markdown(
        """
- The class labels are rule-of-thumb bands, not official standards.
- The model learned from small organic molecules. Salts, metals, polymers and very large molecules fall outside what it knows.
- Real solubility depends on temperature, pH and crystal form, none of which the model sees.
- The typical range shown is the spread among trees. It is a useful warning sign but not a formal confidence interval.
"""
    )

# ---------- tab 5: developer ----------
with tab_dev:
    st.markdown(
        """
<div class="card">
<div class="label">Developer</div>
<div class="big" style="font-size:2rem">Agboola Abdulroheem Oyindamola</div>
<div><b>Computational Chemist and Biochemist</b></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(
        "I am a Biochemistry graduate of the University of Ilorin with First Class Honours, based in Nigeria. "
        "My background covers wet lab research, quality control, data analysis and bioinformatics, and I now focus on using machine learning to predict "
        "the properties of molecules and materials. Esol is one of those projects, built to make solubility prediction quick and easy to understand."
    )
    st.subheader("Other work")
    st.markdown(
        """
- **ThermoPET:** predicts the thermal stability of PETase mutants using ESM-2 embeddings and XGBoost, for plastic recycling.
- **OmniZyme:** predicts enzyme and substrate promiscuity with ESM-2 and ChemBERTa, to help find industrial biocatalysts.
"""
    )
    st.subheader("Get in touch")
    st.markdown(
        """
- Portfolio: [abdulroheemagboola.netlify.app](https://abdulroheemagboola.netlify.app)
- GitHub: [bigthinker-code](https://github.com/bigthinker-code)
- Medium: [@roheemoyindamola](https://medium.com/@roheemoyindamola)
"""
    )
