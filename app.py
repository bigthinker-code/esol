from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Draw, rdFingerprintGenerator, rdMolDescriptors

st.set_page_config(page_title="Esol | Solubility Predictor", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")

BASE = Path(__file__).parent
MODEL_FILE = BASE / "combined_rf_esol_model.joblib"
TEST_FILE = BASE / "combined_rf_test_predictions.csv"
METRICS_FILE = BASE / "model_metrics.csv"

INK, MUTED, LINE = "#10263A", "#5B7183", "#D9E3EA"
WATER, GOOD, BAD = "#1F7A9E", "#2F8F6B", "#C0563B"
ALLOWED = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;500;600&display=swap');
:root {--ink:#10263A; --muted:#5B7183; --line:#D9E3EA; --water:#1F7A9E; --good:#2F8F6B; --bad:#C0563B;}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1080px;}

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp dt, .stApp dd, .stApp button, div[data-testid="stMarkdownContainer"] {font-family: 'Source Sans 3', sans-serif;}
h1, h2, h3, .brand, .logs, .facts dd {font-family: 'Bricolage Grotesque', 'Source Sans 3', sans-serif !important; color: var(--ink);}
.stApp p, .stApp li {line-height: 1.6;}

/* header */
.top {display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap;
      padding-bottom: 1.2rem; margin-bottom: .6rem;}
.brand {font-size: 3rem; font-weight: 700; letter-spacing: -.02em; line-height: 1; margin: 0; padding: 0;}
.tag {font-size: 1.15rem; margin: .4rem 0 0; color: var(--muted);}
.meta {font-size: .92rem; color: var(--muted); margin: 0; max-width: 320px; text-align: right;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 1.8rem; border-bottom: 1px solid var(--line);}
.stTabs button[data-baseweb="tab"] {padding: .7rem 0; background: transparent; height: auto;}
.stTabs button[data-baseweb="tab"] p {font-weight: 600; font-size: 1.02rem; color: var(--muted);}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {color: var(--ink);}
.stTabs [data-baseweb="tab-highlight"] {background: var(--water); height: 3px;}
.stTabs [data-baseweb="tab-border"] {display: none;}

/* inputs and buttons */
div[data-baseweb="input"], div[data-baseweb="base-input"] {border-radius: 10px;}
div[data-baseweb="input"] input {font-size: 1.1rem; padding: .7rem .9rem;}
.stButton > button {border-radius: 999px; border: 1px solid #BFD0DB; background: #fff; color: var(--ink);
                    padding: .1rem .9rem; min-height: 2.1rem; width: 100%;}
.stButton > button:hover {border-color: var(--water); color: var(--water); background: #fff;}
.stButton > button p {font-size: .92rem; font-weight: 500;}
.hint {color: var(--muted); font-size: .92rem; margin: .2rem 0 .5rem;}

/* result */
.result {display: grid; grid-template-columns: minmax(220px, 300px) 1fr; gap: 2.4rem; align-items: center;
         background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 1.8rem 2rem; margin-top: 1.2rem;}
.result .small {font-size: .95rem; color: var(--muted); margin: 0;}
.logs {font-size: 4.2rem; font-weight: 700; line-height: 1; margin: .3rem 0 .2rem; letter-spacing: -.03em;}
.verdict {font-size: 1.35rem; font-weight: 600; margin: 0; color: var(--ink);}
.gauge {padding: 0 1rem;}
.gwrap {position: relative; padding-top: 3rem;}
.bubble {position: absolute; top: 0; transform: translateX(-50%); background: var(--ink); color: #fff;
         font-weight: 600; padding: .15rem .65rem; border-radius: 8px; font-size: .98rem;}
.bubble::after {content: ""; position: absolute; left: 50%; bottom: -5px; margin-left: -5px;
                border: 5px solid transparent; border-bottom: 0; border-top-color: var(--ink);}
.needle {position: absolute; top: 2.1rem; width: 3px; height: 2.35rem; background: var(--ink); transform: translateX(-50%); border-radius: 2px;}
.track {display: flex; height: 22px; border-radius: 11px; overflow: hidden;}
.track span {display: block; height: 100%;}
.ticks, .zones {position: relative; height: 1.3rem; margin-top: .45rem;}
.ticks span, .zones span {position: absolute; transform: translateX(-50%); font-size: .82rem; color: var(--muted); white-space: nowrap;}
.zones span {font-size: .8rem; font-weight: 600;}
@media (max-width: 760px) {.result {grid-template-columns: 1fr; gap: 1.4rem; padding: 1.3rem;} .logs {font-size: 3.4rem;}
                           .meta {text-align: left;}}

/* facts row */
.facts {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem 2rem; margin: 1.6rem 0;
        padding: 1.1rem 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);}
.facts dt {color: var(--muted); font-size: .9rem; margin: 0;}
.facts dd {margin: .15rem 0 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -.01em;}

/* sections */
.sec {font-size: 1.45rem; font-weight: 700; margin: 2.2rem 0 .4rem; letter-spacing: -.01em;}
.note {color: var(--muted); font-size: .93rem; margin: 0 0 .9rem;}
.why {display: grid; gap: .8rem;}
.why div {padding: .15rem 0 .15rem 1rem; border-left: 4px solid var(--line);}
.why .up {border-left-color: var(--good);} .why .down {border-left-color: var(--bad);}
.why b {font-size: 1.02rem;} .why i {font-style: normal; font-size: .85rem; font-weight: 600; margin-left: .5rem;}
.why .up i {color: var(--good);} .why .down i {color: var(--bad);} .why .flat i {color: var(--muted);}
.why p {margin: .1rem 0 0; color: #33495B;}
.molbox {background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: .6rem;}
.dev-name {font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1.1; letter-spacing: -.02em; margin: 0;}
.dev-role {font-size: 1.2rem; color: var(--water); font-weight: 600; margin: .4rem 0 1.2rem;}
.foot {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .88rem;}

/* specificity fixes: Streamlit styles <p> directly */
.stApp p.logs {font-size: 4.4rem; font-weight: 700; line-height: 1; margin: .3rem 0 .2rem; letter-spacing: -.03em; color: var(--ink);}
.stApp p.verdict {font-size: 1.4rem; font-weight: 600; margin: 0; color: var(--ink);}
.stApp p.small {font-size: .95rem; color: var(--muted); margin: 0;}
.stApp p.tag {font-size: 1.15rem; margin: .5rem 0 0; color: var(--muted);}
.stApp p.meta {font-size: .92rem; color: var(--muted); margin: 0;}
.stApp p.hint {font-size: .92rem; color: var(--muted); margin: .3rem 0 .5rem;}
.stApp p.note {font-size: .93rem; color: var(--muted); margin: 0 0 .9rem;}
.stApp p.dev-name {font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin: 0; color: var(--ink);}
.stApp p.dev-role {font-size: 1.2rem; font-weight: 600; color: var(--water); margin: .4rem 0 1.2rem;}
.stApp .why p {font-size: .98rem; margin: .1rem 0 0; color: #33495B;}
.stApp .stTabs button[data-baseweb="tab"] p {font-size: 1.05rem !important; font-weight: 600 !important; color: var(--muted);}
.stApp .stTabs button[data-baseweb="tab"][aria-selected="true"] p {color: var(--ink) !important;}
.stApp h3.sec {font-size: 1.45rem; margin: 2.2rem 0 .4rem; padding: 0;}
@media (max-width: 760px) {.stApp p.logs {font-size: 3.4rem;} .stApp p.meta {text-align: left;}}

.stApp [data-testid="stImage"] img {border: 1px solid var(--line); border-radius: 14px; background: #fff;}
.stApp [data-testid="stExpander"] {margin-top: 1.6rem;}
.stApp h1.brand {padding: 0 !important; margin: 0 !important;}
.stApp div[data-testid="stButton"] button {white-space: nowrap;}
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
        return "Highly soluble", "Roughly 0.1 mol/L or more dissolves, so this compound should go into water with little effort."
    if logs >= -3:
        return "Soluble", "It should dissolve at normal lab concentrations, from about 1 mmol/L up to 0.1 mol/L."
    if logs >= -5:
        return "Poorly soluble", "Only small amounts dissolve. In drug work this range often needs formulation help."
    return "Very poorly soluble", "Almost nothing dissolves in water. Expect it to stay as a solid or an oil."


def confidence_text(std):
    if std < 0.6:
        return "high", "The trees in the forest give very similar answers, so the model is fairly settled on this one."
    if std < 0.9:
        return "moderate", "The trees differ a little. Treat the number as a good estimate, not an exact value."
    return "low", "The trees disagree a lot. This molecule may look unlike what the model learned, so be careful with the result."


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
        pts.append(("down", "Lipophilicity", f"LogP is {lp:.1f}. The molecule is oily and prefers fat-like surroundings to water."))
    elif lp <= 1:
        pts.append(("up", "Lipophilicity", f"LogP is {lp:.1f}. The molecule is water-friendly, which helps it dissolve."))
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
    return mol_l, mol_l * mw


def fmt_sci(v):
    return f"{v:.2e}" if (v < 0.001 or v >= 1000) else f"{v:.3g}"


def gauge_html(pred):
    lo, hi = -8.0, 1.0
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    p = pos(pred)
    bub = min(max(p, 4), 96)
    zones = [(-8, -5, "#F1DCD2", "Very poor"), (-5, -3, "#EFE5CF", "Poor"), (-3, -1, "#D8EBE2", "Soluble"), (-1, 1, "#CDE5F0", "High")]
    track = "".join(f'<span style="width:{pos(b) - pos(a):.2f}%;background:{c}"></span>' for a, b, c, _ in zones)
    names = "".join(f'<span style="left:{(pos(a) + pos(b)) / 2:.2f}%">{n}</span>' for a, b, _, n in zones)
    ticks = "".join(f'<span style="left:{pos(v):.2f}%">{v}</span>' for v in (-8, -6, -4, -2, 0))
    return (
        f'<div class="gauge"><div class="gwrap">'
        f'<div class="bubble" style="left:{bub:.2f}%">{pred:.2f}</div>'
        f'<div class="needle" style="left:{p:.2f}%"></div>'
        f'<div class="track">{track}</div>'
        f'<div class="ticks">{ticks}</div><div class="zones">{names}</div>'
        f"</div></div>"
    )


def style_fig(fig, height=340):
    fig.update_layout(
        template="simple_white", height=height, margin=dict(t=50, l=10, r=10, b=40),
        font=dict(family="Source Sans 3, sans-serif", color=INK, size=14),
        title_font=dict(family="Bricolage Grotesque, sans-serif", size=17),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#E6EDF2", zerolinecolor="#E6EDF2")
    fig.update_yaxes(gridcolor="#E6EDF2", zerolinecolor="#E6EDF2")
    return fig


def section(title, note=None):
    st.markdown(f'<h3 class="sec">{title}</h3>' + (f'<p class="note">{note}</p>' if note else ""), unsafe_allow_html=True)


def facts(items):
    html = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items)
    st.markdown(f'<dl class="facts">{html}</dl>', unsafe_allow_html=True)


# ---------- header ----------
meta = ""
if metrics_df is not None and test_df is not None:
    b0 = metrics_df.iloc[0]
    meta = f"Random Forest with R² {b0['R2']:.2f}. Average error {b0['MAE']:.2f} log units on {len(test_df)} molecules it had never seen."
st.markdown(
    f"""
<div class="top">
  <div><h1 class="brand">Esol</h1><p class="tag">How well does a molecule dissolve in water?</p></div>
  <p class="meta">{meta}</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_perf, tab_about, tab_dev = st.tabs(["Predict", "Batch", "Model performance", "How it works", "About the developer"])

# ---------- predict ----------
with tab_predict:
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = EXAMPLES["Aspirin"]

    def set_example(smi):
        st.session_state.smiles_input = smi

    st.text_input("Molecule as a SMILES string", key="smiles_input", placeholder="For example CC(=O)Oc1ccccc1C(=O)O",
                  help="SMILES is a line of text that describes a molecule. Press Enter to predict.")
    st.markdown('<p class="hint">Or try one of these</p>', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 1.35, 1.35, 0.6])
    for col, (name, smi) in zip(cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}", on_click=set_example, args=(smi,))

    smiles = st.session_state.smiles_input.strip()
    res = featurize(smiles)

    if not smiles:
        st.info("Type or paste a SMILES string above to see a prediction.")
    elif res is None:
        st.error("That SMILES could not be read. Check for typos and for unclosed rings or brackets.")
    else:
        X = res["x"].reshape(1, -1)
        mean, std = predict_matrix(X)
        pred, spread = float(mean[0]), float(std[0])
        mol_l, g_l = solubility_units(pred, res["desc"]["MolWt"])
        cls, cls_text = solubility_class(pred)
        conf, conf_text = confidence_text(spread)

        for f in input_flags(res["mol"], smiles):
            st.warning(f)

        st.markdown(
            f'<div class="result"><div><p class="small">Predicted logS</p><p class="logs">{pred:.2f}</p>'
            f'<p class="verdict">{cls}</p></div>{gauge_html(pred)}</div>',
            unsafe_allow_html=True,
        )
        facts([
            ("Solubility in mol/L", fmt_sci(mol_l)),
            ("Solubility in mg/mL", fmt_sci(g_l)),
            ("Likely range (logS)", f"{pred - spread:.1f} to {pred + spread:.1f}"),
        ])

        st.write(cls_text)
        conf_color = {"high": GOOD, "moderate": "#B7791F", "low": BAD}[conf]
        st.markdown(f'Model agreement is <b style="color:{conf_color}">{conf}</b>. {conf_text}', unsafe_allow_html=True)
        if test_df is not None:
            w05 = (test_df["residual"].abs() <= 0.5).mean() * 100
            w10 = (test_df["residual"].abs() <= 1.0).mean() * 100
            st.markdown(
                f'<p class="note">On the test set, {w05:.0f}% of predictions landed within 0.5 log units of the measured value and {w10:.0f}% within 1.0. '
                f"One log unit is a tenfold change in solubility, so 0.5 is about a threefold gap. The likely range above held the measured value for about 78% of test molecules.</p>",
                unsafe_allow_html=True,
            )

        left, right = st.columns([1, 1.35], gap="large")
        with left:
            section("Structure")
            st.image(Draw.MolToImage(res["mol"], size=(460, 340)))
        with right:
            section("Why it behaves this way", "General chemistry trends read from the molecule's properties, not the model's internal reasoning.")
            labels = {"up": "Helps solubility", "down": "Hurts solubility", "flat": "Neutral"}
            items = "".join(f'<div class="{d}"><b>{t}</b><i>{labels[d]}</i><p>{x}</p></div>' for d, t, x in explain_factors(res["desc"], res["aromatic_rings"]))
            st.markdown(f'<div class="why">{items}</div>', unsafe_allow_html=True)

        with st.expander("Calculated descriptors"):
            dd = pd.DataFrame({"Descriptor": list(res["desc"].keys()), "Value": [round(v, 3) for v in res["desc"].values()]})
            st.dataframe(dd, hide_index=True)

        if test_df is not None:
            section("Similar molecules with known solubility",
                    "Taken from the held-out test set. Similarity is the Tanimoto score, where 1.0 means identical. Low scores mean no close relatives were found.")
            sims = DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(res["mol"]), test_fingerprints())
            top = np.argsort(sims)[::-1][:5]
            near = test_df.iloc[top][["smiles", "logS", "predicted_logS"]].copy()
            near.insert(1, "Similarity", [round(sims[i], 2) for i in top])
            near.columns = ["SMILES", "Similarity", "Measured logS", "Model logS"]
            st.dataframe(near, hide_index=True)

            fig = go.Figure()
            fig.add_histogram(x=test_df["logS"], nbinsx=30, marker_color="rgba(31,122,158,.45)")
            fig.add_vline(x=pred, line_width=3, line_color=INK, annotation_text="Your molecule", annotation_position="top")
            fig.update_layout(title="Where your molecule sits among the test compounds", xaxis_title="logS", yaxis_title="Count", showlegend=False)
            st.plotly_chart(style_fig(fig, 300), theme=None)

# ---------- batch ----------
with tab_batch:
    section("Predict many molecules at once", "Upload a CSV with a column named smiles, or paste one SMILES per line.")
    up = st.file_uploader("CSV file", type="csv")
    pasted = st.text_area("Or paste SMILES here", height=120, placeholder="CCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O")

    smiles_list = []
    if up is not None:
        raw = pd.read_csv(up)
        col = next((c for c in raw.columns if c.strip().lower() == "smiles"), None)
        if col is None:
            st.error("No column named smiles was found in that file. Rename your SMILES column and upload it again.")
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
        st.caption(f"{len(ok)} predicted, {len(smiles_list) - len(ok)} could not be read.")
        st.download_button("Download results as CSV", out.to_csv(index=False).encode(), "esol_predictions.csv", "text/csv")
        if ok:
            fig = go.Figure(go.Histogram(x=out["predicted_logS"].dropna(), nbinsx=25, marker_color="rgba(31,122,158,.55)"))
            fig.update_layout(title="Distribution of predicted logS", xaxis_title="Predicted logS", yaxis_title="Count")
            st.plotly_chart(style_fig(fig, 300), theme=None)

# ---------- performance ----------
with tab_perf:
    if metrics_df is None or test_df is None:
        st.info("Add model_metrics.csv and combined_rf_test_predictions.csv to the app folder to see this tab.")
    else:
        best = metrics_df.iloc[0]
        section("How well does it work?")
        facts([("R²", f"{best['R2']:.3f}"), ("Mean absolute error", f"{best['MAE']:.3f}"), ("Root mean squared error", f"{best['RMSE']:.3f}")])
        with st.expander("What do these numbers mean?"):
            st.write(f"R² shows how much of the variation in solubility the model captures. {best['R2']:.2f} means about {best['R2'] * 100:.0f}%, where 1.0 is perfect and 0 is no better than guessing the average.")
            st.write(f"Mean absolute error is the average size of the miss, {best['MAE']:.2f} log units here. Root mean squared error is similar but punishes big misses harder, and at {best['RMSE']:.2f} it shows that a few molecules are badly off.")

        section("Measured against predicted", "Points on the dashed line are perfect. The model tracks the trend well and is least accurate for the most soluble compounds, where a few points sit far below the line.")
        lo = float(min(test_df["logS"].min(), test_df["predicted_logS"].min()))
        hi = float(max(test_df["logS"].max(), test_df["predicted_logS"].max()))
        fig = go.Figure()
        fig.add_scatter(x=test_df["logS"], y=test_df["predicted_logS"], mode="markers", text=test_df["smiles"],
                        marker=dict(size=8, color="rgba(31,122,158,.6)"), name="Test molecules",
                        hovertemplate="%{text}<br>Measured %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>")
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=BAD, dash="dash"), name="Perfect prediction")
        fig.update_layout(xaxis_title="Measured logS", yaxis_title="Predicted logS", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(style_fig(fig, 480), theme=None)

        l, r = st.columns(2, gap="large")
        with l:
            fig = go.Figure(go.Histogram(x=test_df["residual"], nbinsx=30, marker_color="rgba(31,122,158,.55)"))
            fig.update_layout(title="Errors (measured minus predicted)", xaxis_title="Error")
            st.plotly_chart(style_fig(fig, 320), theme=None)
            st.caption("A bell shape centred near zero means the model is not biased high or low overall.")
        with r:
            fig = go.Figure(go.Bar(x=metrics_df["Model"], y=metrics_df["MAE"], marker_color="rgba(31,122,158,.7)"))
            fig.update_layout(title="Error by model (lower is better)", yaxis_title="MAE")
            st.plotly_chart(style_fig(fig, 320), theme=None)
            st.caption("The tuned Random Forest beat the stacked model, the descriptor-only forest, a Ridge model and the plain average.")

        section("What the model relies on")
        imp = model.feature_importances_
        d_imp = pd.Series(imp[: len(DESC_COLS)], index=DESC_COLS).sort_values()
        fp_total = float(imp[len(DESC_COLS):].sum())
        fig = go.Figure(go.Bar(x=d_imp.values, y=d_imp.index, orientation="h", marker_color="rgba(31,122,158,.7)"))
        fig.update_layout(title="Importance of each descriptor")
        st.plotly_chart(style_fig(fig, 340), theme=None)
        st.caption(f"The 2048 fingerprint bits together account for {fp_total * 100:.0f}% of total importance. Importance shows how much the forest used a feature, not whether it raises or lowers solubility.")
        with st.expander("Full comparison table"):
            st.dataframe(metrics_df.round(3), hide_index=True)

# ---------- how it works ----------
with tab_about:
    section("From structure to prediction")
    st.markdown(
        """
1. **SMILES in.** A SMILES string is a line of text that describes a molecule's structure, such as `CCO` for ethanol.
2. **Features out.** RDKit reads the structure and calculates eight descriptors: molecular weight, LogP, TPSA, hydrogen bond donors and acceptors, rotatable bonds, ring count and the fraction of sp3 carbons. It also builds a 2048-bit Morgan fingerprint that records which small substructures are present.
3. **Forest votes.** A Random Forest is a large group of decision trees. Each tree makes its own prediction and the average becomes the answer. The spread between trees is the agreement indicator.
4. **Translate.** The logS value is converted to mol/L and mg/mL and sorted into a plain-language class.
"""
    )
    section("Reading logS", "logS is the base-10 logarithm of solubility in mol/L. A logS of -2 means 0.01 mol/L and -4 means 0.0001 mol/L. Each step down is ten times less soluble.")
    st.table(pd.DataFrame({
        "logS": ["-1 or higher", "-1 to -3", "-3 to -5", "Below -5"],
        "Class": ["Highly soluble", "Soluble", "Poorly soluble", "Very poorly soluble"],
        "Rough meaning": ["0.1 mol/L or more", "About 1 mmol/L to 0.1 mol/L", "About 10 µmol/L to 1 mmol/L", "Below about 10 µmol/L"],
    }))
    section("Limits to keep in mind")
    st.markdown(
        """
- The class labels are rule-of-thumb bands, not official standards.
- The model learned from small organic molecules. Salts, metals, polymers and very large molecules fall outside what it knows.
- Real solubility depends on temperature, pH and crystal form, none of which the model sees.
- The likely range is the spread among trees. It is a useful warning sign but not a formal confidence interval.
"""
    )

# ---------- developer ----------
with tab_dev:
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        st.markdown(
            '<p class="dev-name">Agboola Abdulroheem Oyindamola</p><p class="dev-role">Computational Chemist and Biochemist</p>',
            unsafe_allow_html=True,
        )
        st.write(
            "I am a Biochemistry graduate of the University of Ilorin with First Class Honours, based in Nigeria. "
            "My background covers wet lab research, quality control, data analysis and bioinformatics, and I now focus on using machine learning to predict "
            "the properties of molecules and materials. Esol is one of those projects, built to make solubility prediction quick and easy to understand."
        )
        section("Other work")
        st.markdown(
            """
- **ThermoPET** predicts the thermal stability of PETase mutants using ESM-2 embeddings and XGBoost, for plastic recycling.
- **OmniZyme** predicts enzyme and substrate promiscuity with ESM-2 and ChemBERTa, to help find industrial biocatalysts.
"""
        )
    with b:
        section("Get in touch")
        st.markdown(
            """
- Portfolio: [abdulroheemagboola.netlify.app](https://abdulroheemagboola.netlify.app)
- GitHub: [bigthinker-code](https://github.com/bigthinker-code)
- Medium: [@roheemoyindamola](https://medium.com/@roheemoyindamola)
"""
        )

st.markdown(
    '<div class="foot">Esol, built by Agboola Abdulroheem Oyindamola. A research tool that does not replace lab measurement.</div>',
    unsafe_allow_html=True,
)
