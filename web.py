import io
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Gemini is optional: the app still works with only the CNN if the key/package is missing.
try:
    from gemini_helper import get_advice, gemini_available
except Exception:
    get_advice = None

    def gemini_available():
        return False


st.set_page_config(
    page_title="AgriGuard AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

HOME, DETECT, ABOUT = "🏠 Home", "🔬 Detect", "📖 About"

CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Huanglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold',
    'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy',
]

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Source+Sans+3:wght@400;600&display=swap');

:root{
  --ink:#14301f; --leaf:#2f6b3f; --sage:#eef3e6; --paper:#fbfcf7;
  --marigold:#f2a007; --soil:#6b4423; --line:#d5dfc8; --alert:#b3261e;
}

/* remove default chrome + sidebar */
#MainMenu, header, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}

.stApp{background:var(--sage);}
.block-container{max-width:860px;padding:2rem 1.2rem 8rem;}

.stApp, .stApp p, .stApp li, .stApp label, .stApp div[data-testid="stCaptionContainer"]{
  color:var(--ink); font-family:'Source Sans 3',system-ui,sans-serif;
}
.stApp span{color:var(--ink);}
/* keep Streamlit's icon font (expander arrows etc.) */
.stApp [data-testid="stIconMaterial"], .stApp [data-testid="stIconMaterial"] *,
.stApp [class*="material-symbols"], .stApp [class*="material-icons"]{
  font-family:'Material Symbols Rounded','Material Icons' !important;
  font-feature-settings:'liga' !important; letter-spacing:normal !important;
}

/* explicit widget colors so nothing depends on the viewer's light/dark theme */
:root{color-scheme:light;}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stBaseButton-secondary"]{
  background:var(--paper) !important;border:1px solid var(--leaf) !important;border-radius:10px !important;}
[data-testid="stFileUploaderDropzone"] button *, [data-testid="stBaseButton-secondary"] *{color:var(--ink) !important;}
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stBaseButton-secondary"]:hover{
  background:var(--leaf) !important;}
[data-testid="stFileUploaderDropzone"] button:hover *, [data-testid="stBaseButton-secondary"]:hover *{color:#fff !important;}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{color:var(--ink) !important;}
[data-testid="stAlert"]{background:#fff !important;border:1px solid var(--line) !important;
  border-left:6px solid var(--marigold) !important;border-radius:12px !important;}
[data-testid="stAlert"] *{color:var(--ink) !important;}
[data-baseweb="select"] > div{background:var(--paper) !important;border:1px solid var(--line) !important;}
[data-baseweb="select"] *{color:var(--ink) !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4{
  color:var(--ink); font-family:'Bricolage Grotesque',system-ui,sans-serif; letter-spacing:-0.01em;
}

/* hero: the one bold moment */
.hero{background:var(--ink);border-radius:22px;padding:2.4rem 2rem 2.2rem;margin-bottom:1rem;}
.stApp .hero h1{color:#f4f7ea;font-size:2.7rem;line-height:1.08;margin:0 0 .8rem;font-weight:800;}
.stApp .hero p{color:#cfdcc4;font-size:1.12rem;max-width:34rem;margin:0;}

/* stats strip */
.strip{display:flex;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:1.4rem 0;}
.strip div{flex:1 1 120px;padding:.9rem 1rem;border-left:1px solid var(--line);}
.strip div:first-child{border-left:none;padding-left:0;}
.strip b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.7rem;color:var(--leaf);}
.strip span{font-size:.92rem;}

/* steps (a real sequence) */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;margin:.6rem 0 1rem;}
.step{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:1rem;}
.step b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.05rem;display:block;margin-bottom:.25rem;}
.step i{font-style:normal;color:var(--soil);font-weight:600;}

/* result card */
.result{background:var(--paper);border:1px solid var(--line);border-left:8px solid var(--alert);
        border-radius:14px;padding:1.2rem 1.4rem;margin:1rem 0;}
.result.ok{border-left-color:var(--leaf);}
.result small{color:var(--soil);font-weight:600;}
.stApp .result h2{margin:.15rem 0 .5rem;font-size:1.9rem;}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;}
.bar div{height:100%;background:var(--leaf);}

/* widgets */
[data-testid="stFileUploaderDropzone"]{background:var(--paper);border:2px dashed #7fa47f;border-radius:14px;}
[data-testid="stExpander"]{background:var(--paper);border:1px solid var(--line);border-radius:12px;}
[data-testid="stImage"] img{border-radius:16px;}
button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--marigold) !important;border:none !important;border-radius:10px !important;font-weight:700 !important;}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p{color:var(--ink) !important;}

/* bottom dock (replaces sidebar) */
.st-key-bottom_nav{
  position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:1000;width:max-content;max-width:94vw;
  background:var(--ink);border-radius:999px;padding:6px;box-shadow:0 8px 30px rgba(20,48,31,.35);
}
.st-key-bottom_nav [role="radiogroup"]{gap:2px;flex-wrap:nowrap;justify-content:center;}
.st-key-bottom_nav label[data-baseweb="radio"]{margin:0;padding:9px 22px;border-radius:999px;cursor:pointer;}
.st-key-bottom_nav label[data-baseweb="radio"] > div:first-child{display:none;}
.st-key-bottom_nav label[data-baseweb="radio"] p{color:#cfdcc4;font-weight:600;white-space:nowrap;}
.st-key-bottom_nav label[data-baseweb="radio"]:has(input:checked){background:var(--marigold);}
.st-key-bottom_nav label[data-baseweb="radio"]:has(input:checked) p{color:var(--ink);font-weight:800;}

@media (max-width:640px){
  .stApp .hero h1{font-size:2rem;}
  .st-key-bottom_nav label[data-baseweb="radio"]{padding:9px 14px;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")


def model_prediction(image_bytes):
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(image_bytes), target_size=(128, 128))
    arr = np.array([tf.keras.preprocessing.image.img_to_array(image)])
    prediction = load_model().predict(arr, verbose=0)
    return int(np.argmax(prediction)), float(np.max(prediction)) * 100


# ---------------------------------------------------------------- navigation
with st.container(key="bottom_nav"):
    page = st.radio("Navigate", [HOME, DETECT, ABOUT], horizontal=True,
                    key="page", label_visibility="collapsed")


def go_detect():
    st.session_state["page"] = DETECT


# ---------------------------------------------------------------- pages
def render_home():
    st.markdown("""
    <div class="hero">
      <h1>Catch crop disease before it spreads.</h1>
      <p>Photograph a leaf and AgriGuard AI names the problem in seconds.
         With Gemini, it also explains what to do next in your own language.</p>
    </div>
    """, unsafe_allow_html=True)
    st.button("Check a leaf now", type="primary", on_click=go_detect)

    st.markdown("""
    <div class="strip">
      <div><b>38</b><span>conditions recognised</span></div>
      <div><b>14</b><span>crops covered</span></div>
      <div><b>98.7%</b><span>validation accuracy</span></div>
      <div><b>&lt; 5 s</b><span>per leaf</span></div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("homeIMG.jpg"):
        st.image("homeIMG.jpg", use_container_width=True, caption="Healthy crops, better harvest")

    st.subheader("How it works")
    st.markdown("""
    <div class="steps">
      <div class="step"><i>1</i><b>Photograph</b>Take a clear photo of one affected leaf.</div>
      <div class="step"><i>2</i><b>Upload</b>Open Detect and add the photo.</div>
      <div class="step"><i>3</i><b>Analyze</b>The CNN identifies the crop and condition.</div>
      <div class="step"><i>4</i><b>Act</b>Gemini explains treatment and prevention.</div>
    </div>
    """, unsafe_allow_html=True)


def render_result(res):
    plant, disease = CLASS_NAMES[res["idx"]].split(" - ", 1)
    healthy = "Healthy" in disease
    cls = "result ok" if healthy else "result"
    head = f"{plant} looks healthy" if healthy else disease
    st.markdown(f"""
    <div class="{cls}">
      <small>{plant}</small>
      <h2>{head}</h2>
      <div class="bar"><div style="width:{res['conf']:.0f}%"></div></div>
      <small>Model confidence {res['conf']:.1f}%</small>
    </div>
    """, unsafe_allow_html=True)

    if res["conf"] < 60:
        st.warning("Low confidence. Try a sharper, closer photo of a single leaf in daylight.")

    advice = res.get("advice")
    if advice is None:
        if not gemini_available():
            st.caption("Treatment advice is off. Add GEMINI_API_KEY to enable it.")
        return
    if "error" in advice:
        st.warning("AI advice is unavailable right now: " + str(advice["error"])[:160])
        return

    if not advice.get("is_leaf", True):
        st.warning("Gemini does not see a plant leaf in this photo. Upload a clear leaf image.")
    elif not advice.get("agrees_with_cnn", True):
        st.warning("Gemini reads this leaf differently from the model. Please confirm with an agriculture expert.")

    st.subheader("What to do next")
    st.write(advice.get("explanation", ""))
    st.markdown(f"**Severity:** {advice.get('severity', '-')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Organic treatment**")
        st.write(advice.get("organic_treatment", "-"))
    with c2:
        st.markdown("**Chemical treatment**")
        st.write(advice.get("chemical_treatment", "-"))
    st.markdown("**Prevention**")
    st.write(advice.get("prevention", "-"))
    st.caption(advice.get("warning") or
               "AI-generated advice. Confirm with your local agriculture officer before spraying.")


def render_detect():
    st.header("Check a leaf")
    st.caption("Use a clear photo of one leaf in good light.")
    ai_on = gemini_available()

    source = st.radio("Photo source", ["📁 Upload photo", "📷 Take photo"], horizontal=True,
                      key="photo_source", label_visibility="collapsed")
    if source.startswith("📁"):
        file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        file = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

    if not file:
        st.info("Add a leaf photo to begin.")
        return

    data = file.getvalue()
    fid = f"{getattr(file, 'name', 'camera')}-{len(data)}"

    left, right = st.columns(2)
    with left:
        st.image(data, use_container_width=True)
    with right:
        lang = st.selectbox("Advice language", ["Hindi", "English", "Marathi"]) if ai_on else "English"
        go = st.button("Analyze leaf", type="primary", use_container_width=True)

    if go:
        with st.spinner("Reading the leaf..."):
            idx, conf = model_prediction(data)
            advice = None
            if ai_on and get_advice:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                advice = get_advice(pil, CLASS_NAMES[idx], conf, lang)
        st.session_state["result"] = {"fid": fid, "idx": idx, "conf": conf, "advice": advice}

    res = st.session_state.get("result")
    if res and res["fid"] == fid:
        render_result(res)


def render_about():
    st.header("About AgriGuard AI")
    st.write("An AI tool that helps farmers identify plant diseases from a leaf photo, "
             "so they can act early and reduce crop losses.")

    with st.expander("Dataset", expanded=True):
        st.markdown("""
- Source: [Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 87,000+ RGB images across 38 classes, 256x256 pixels
- Training split: 70,295 images (80%), validation split: 17,572 images (20%)
- Test set: 33 curated real-world images
- Augmentation: rotation, flipping and zoom
        """)
    with st.expander("Technical architecture"):
        st.markdown("""
- Framework: TensorFlow 2.0
- Model: custom 16-layer CNN, trained for 50 epochs with the Adam optimizer
- Validation accuracy: 98.7%
- Advice layer: Gemini API explains the diagnosis, checks the photo, and suggests treatment
        """)
    st.caption("© 2025 AgriGuard AI. Developed by Rohit in Pune.")


{HOME: render_home, DETECT: render_detect, ABOUT: render_about}[page]()
