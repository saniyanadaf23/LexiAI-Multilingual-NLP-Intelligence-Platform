
import requests
import streamlit as st

emails = st.query_params.get("email", [])
user_email = emails[0] if emails else None

if not user_email:
    st.info("⚠ Demo mode: analysis will not be saved to backend.")

# Reset page session state only if not already set
if "page" not in st.session_state:
    st.session_state.page = "home"
if "view_history_item" not in st.session_state:
    st.session_state.view_history_item = None
import stanza
import pandas as pd
import os
from textblob import TextBlob
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sumy.summarizers.lex_rank import LexRankSummarizer
import plotly.graph_objects as go
import language_tool_python


def save_result_to_db(email, text, feature, result):
    """Send the analysis result to Flask backend for storage."""
    try:
        requests.post("http://localhost:5000/save_analysis", json={
            "email": email,
            "text": text,
            "feature": feature,
            "result": result
        })
    except Exception as e:
        print("⚠️ Could not save analysis:", e)

#from auth import api_handler
#st.query_params = {}


#if "signup_api" in st.session_state:
   # st.write(api_handler())

# Try importing speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Supported languages for Stanza POS tagging
LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur"
}

# Google Speech API language codes
GOOGLE_SPEECH_LANG = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Urdu": "ur-IN"
}

# Hindi translations for POS tags
POS_TAGS_HINDI = {
    "NOUN": "संज्ञा",
    "PROPN": "विशेष नाम",
    "VERB": "क्रिया",
    "ADJ": "विशेषण",
    "ADV": "क्रिया विशेषण",
    "PRON": "सर्वनाम",
    "ADP": "संबंधबोधक",
    "DET": "निर्देशक",
    "AUX": "सहायक क्रिया",
    "CCONJ": "समुच्चयबोधक",
    "SCONJ": "उपवाक्यबोधक",
    "PART": "अविकारी शब्द",
    "INTJ": "विस्मयादिबोधक",
    "NUM": "संख्या",
    "PUNCT": "विराम चिह्न"
}

# Stanza resource directory
os.environ["STANZA_RESOURCES_DIR"] = os.path.expanduser("~\\stanza_resources")

SUPPORTED_LANGS = ["en", "hi", "mr", "ta", "te", "ur"]

@st.cache_resource
def init_stanza(lang_code: str):
    """Initialize Stanza pipeline for POS tagging."""
    if lang_code not in SUPPORTED_LANGS:
        st.warning(f"⚠ POS tagging is not available for '{lang_code}'.")
        return None
    models_dir = os.path.join(os.getcwd(), "stanza_models")
    os.environ["STANZA_RESOURCES_DIR"] = models_dir
    try:
        stanza.download(lang_code, verbose=False)
    except Exception:
        pass
    return stanza.Pipeline(lang=lang_code, processors="tokenize,pos", use_gpu=False, verbose=False)

def transcribe_from_mic(language):
    """Record and transcribe audio from microphone."""
    if not SR_AVAILABLE:
        st.error("SpeechRecognition not installed or PyAudio missing.")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now! Recording for a few seconds...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=5, phrase_time_limit=8)
    try:
        st.info("🧠 Transcribing your voice...")
        text = r.recognize_google(audio, language=GOOGLE_SPEECH_LANG.get(language, "en-IN"))
        st.success("✅ Transcription complete!")
    except sr.UnknownValueError:
        text = ""
        st.warning("Could not understand your voice clearly. Try again.")
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        text = ""
    return text

def check_grammar(text, language):
    """Hybrid Grammar Correction: LanguageTool + AI-based refinement"""
    import language_tool_python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    import pandas as pd

    if language != "English":
        return text, pd.DataFrame([{"Message": "⚠ Grammar correction is available only for English."}])

    # Step 1: Basic correction using LanguageTool
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    basic_corrected = language_tool_python.utils.correct(text, matches)

    # Step 2: Deep AI-based grammar refinement
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

    input_ids = tokenizer.encode(basic_corrected, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    deep_corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Step 3: Merge results
    corrected = deep_corrected.strip()

    # Build error table
    errors = []
    for m in matches:
        errors.append({
            "Issue": m.ruleId,
            "Message": m.message,
            "Incorrect Text": m.context,
            "Suggestion": ", ".join(m.replacements) if m.replacements else "-"
        })
    df = pd.DataFrame(errors)
    return corrected, df


    errors_df = pd.DataFrame(issues)
    return corrected_text, errors_df
    error_list = [{"Issue": m.ruleId, "Message": m.message, "Incorrect Text": m.context,
                   "Suggestion": ", ".join(m.replacements) if m.replacements else "-"} for m in matches]
    return corrected, pd.DataFrame(error_list)

def stanza_pos(nlp, text):
    doc = nlp(text)
    rows = [(w.text, w.upos, w.xpos if hasattr(w, "xpos") else "_") for sent in doc.sentences for w in sent.words]
    return rows

def render_tagged_html(rows, show_hindi_tags=False, language="English"):
    color_map = {
        "NOUN": "#e3f2fd", "PROPN": "#fce4ec", "VERB": "#e8f5e8", "ADJ": "#fff3e0",
        "ADV": "#f3e5f5", "PRON": "#ede7f6", "ADP": "#e0f2f1", "NUM": "#fafafa",
        "PUNCT": "#f5f5f5", "DET": "#ffebee", "AUX": "#e8f5e8", "PART": "#f1f8e9",
    }
    html_parts = []
    for token, upos, xpos in rows:
        color = color_map.get(upos, "#f5f5f5")
        label = POS_TAGS_HINDI.get(upos, upos) if show_hindi_tags and language == "Hindi" else upos
        # Improved contrast: Black text for visibility
        html_parts.append(f"<span style='background:{color}; color: #333; padding:4px 8px; margin:2px; border-radius:6px; display:inline-block; border:1px solid #ccc; font-weight: bold; transition: all 0.3s;'>"
                          f"{token}<br/><small style='color:#555; font-weight: normal;'>{label}</small></span>")
    return " ".join(html_parts)

# --- Streamlit UI ---
st.set_page_config(page_title="LinguoAI", layout="wide", page_icon="🌐")


# =======================================
# 🌟 PROFESSIONAL UI/UX DESIGN ENHANCEMENTS
# =======================================

st.set_page_config(page_title="LexiAI — Multilingual NLP Intelligence Platform", page_icon="🧠", layout="wide")
# Always use dark mode theme for LexiAI UI
dark_mode = True

st.markdown("""
<style>
/* ====================================== */
/* LEXIAI — Unified Glass + Teal Aesthetic */
/* ====================================== */

/* Base font and smoothing */
html, body, [class*="css"] {
  font-family: 'Poppins', sans-serif !important;
  scroll-behavior: smooth;
  background: radial-gradient(circle at center, #001f1d, #002a26, #001513);
  color: #e5fdf8 !important;
}

/* ============================================ */
/* 🧭 LexiAI Sidebar — Professional & Minimalist */
/* ============================================ */

[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, rgba(13, 48, 47, 0.9), rgba(8, 23, 22, 0.95));
    color: #e2e8f0;
    border-right: 1px solid rgba(94,234,212,0.1);
    box-shadow: 4px 0 12px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
    padding: 1rem 1.2rem;
    font-family: 'Poppins', sans-serif;
}

/* Headings */
[data-testid="stSidebar"] h2 {
    color: #5eead4 !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    text-align: left;
}
[data-testid="stSidebar"] h4, [data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Subsection spacing */
[data-testid="stSidebar"] hr {
    border: 0;
    height: 1px;
    background: rgba(94,234,212,0.15);
    margin: 12px 0;
}

/* Language Dropdown (Professional Glass Look) */
div[data-baseweb="select"] > div {
    background: rgba(13, 48, 47, 0.7) !important;
    border: 1px solid rgba(94,234,212,0.25) !important;
    border-radius: 8px !important;
    color: #d1fae5 !important;
    font-weight: 500 !important;
    transition: all 0.25s ease-in-out;
}
div[data-baseweb="select"] > div:hover {
    border-color: #5eead4 !important;
    box-shadow: 0 0 10px rgba(94,234,212,0.25);
}

/* Dropdown popup */
div[role="listbox"] {
    background: rgba(11, 38, 37, 0.95) !important;
    border: 1px solid rgba(94,234,212,0.25);
    border-radius: 10px;
    backdrop-filter: blur(10px);
}
div[role="option"] {
    color: #e2e8f0 !important;
    transition: all 0.2s ease;
}
div[role="option"]:hover {
    background: rgba(45, 212, 191, 0.2) !important;
    color: #5eead4 !important;
}

/* Radio & Checkbox - Professional Look */
div[role="radiogroup"] label, 
div[role="checkbox"] label {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #e5e7eb !important;
    font-weight: 500;
    background: transparent;
    padding: 4px 8px;
    border-radius: 6px;
    transition: all 0.2s ease;
}

/* Hover glow */
div[role="radiogroup"] label:hover, 
div[role="checkbox"] label:hover {
    background: rgba(45, 212, 191, 0.08);
    color: #5eead4 !important;
    transform: translateX(3px);
}

/* Checked indicator subtle glow */
input[type="radio"]:checked + div,
input[type="checkbox"]:checked + div {
    color: #5eead4 !important;
    font-weight: 600 !important;
}

/* Section labels */
.st-emotion-cache-1v0mbdj {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 10px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background-color: rgba(94,234,212,0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background-color: rgba(94,234,212,0.5);
}

/* Glow animation on hover for all inputs */
[data-testid="stSidebar"] *:focus, 
[data-testid="stSidebar"] *:hover {
    outline: none;
}

/* Footer inside sidebar */
.sidebar-footer {
    margin-top: 20px;
    text-align: center;
    font-size: 0.8rem;
    color: #64748b;
}
.sidebar-footer b {
    color: #5eead4;
}

/* ============================================= */
/* 🧠 LexiAI — Radio & Checkbox Button Styling */
/* ============================================= */

/* Hide default radio/checkbox inputs */
div[role="radiogroup"] input[type="radio"],
div[role="checkbox"] input[type="checkbox"] {
    display: none !important;
}

/* Button-style layout */
div[role="radiogroup"] label,
div[role="checkbox"] label {
    display: inline-block;
    width: 100%;
    background: rgba(15, 35, 32, 0.7);
    border: 1px solid rgba(94,234,212,0.2);
    border-radius: 8px;
    color: #d1fae5;
    text-align: center;
    font-weight: 500;
    padding: 10px 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.25s ease-in-out;
    box-shadow: inset 0 0 6px rgba(20,184,166,0.15);
}

/* Hover Glow */
div[role="radiogroup"] label:hover,
div[role="checkbox"] label:hover {
    background: rgba(45, 212, 191, 0.15);
    border-color: rgba(94,234,212,0.5);
    box-shadow: 0 0 10px rgba(94,234,212,0.35);
    transform: translateY(-2px);
}

/* Active / Selected Button Look */
div[role="radiogroup"] input[type="radio"]:checked + div,
div[role="checkbox"] input[type="checkbox"]:checked + div {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    color: #ffffff !important;
    border: 1px solid #5eead4 !important;
    box-shadow: 0 0 12px rgba(45, 212, 191, 0.6);
    font-weight: 600;
}

/* Smooth transitions */
div[role="radiogroup"] label, 
div[role="checkbox"] label {
    transition: all 0.25s ease-in-out;
}

/* Optional: subtle ripple when clicked */
div[role="radiogroup"] label:active,
div[role="checkbox"] label:active {
    transform: scale(0.98);
    box-shadow: 0 0 15px rgba(94,234,212,0.25);
}


/* ============================== */
/* 🌐 TAB BAR — NEON GLOW EFFECT */
/* ============================== */

/* General tab styling */
button[data-baseweb="tab"] {
  color: #b9fdf0 !important;
  font-weight: 600 !important;
  background: transparent !important;
  border: none !important;
  padding: 12px 20px !important;
  border-radius: 8px;
  position: relative;
  transition: all 0.25s ease;
  text-shadow: 0 0 5px rgba(94,234,212,0.3);
}

/* Neon underline animation */
button[data-baseweb="tab"]::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%) scaleX(0);
  width: 60%;
  height: 2.5px;
  background: linear-gradient(90deg, #34d399, #5eead4);
  border-radius: 2px;
  transition: transform 0.3s ease;
}

/* Hover glow */
button[data-baseweb="tab"]:hover {
  color: #5eead4 !important;
  text-shadow: 0 0 8px rgba(94,234,212,0.6);
  transform: translateY(-2px);
}

/* Hover underline expand */
button[data-baseweb="tab"]:hover::after {
  transform: translateX(-50%) scaleX(1);
}

/* Active tab — stays glowing */
button[data-baseweb="tab"][aria-selected="true"] {
  color: #34f6c4 !important;
  text-shadow: 0 0 12px rgba(20,184,166,0.8);
}

button[data-baseweb="tab"][aria-selected="true"]::after {
  transform: translateX(-50%) scaleX(1);
  box-shadow: 0 0 12px rgba(94,234,212,0.6);
}

/* Small glow on all tab group container */
[data-baseweb="tab-list"] {
  background: rgba(0, 35, 33, 0.5);
  border-radius: 12px;
  backdrop-filter: blur(10px);
  padding: 8px 10px;
  border: 1px solid rgba(94,234,212,0.15);
  box-shadow: inset 0 0 10px rgba(94,234,212,0.05);
}

/* HERO CARD */
.hero {
  background: linear-gradient(145deg, rgba(0,64,59,0.9), rgba(0,90,78,0.9));
  border-radius: 20px;
  text-align: center;
  padding: 60px 40px;
  margin-top: 30px;
  box-shadow: 0 0 40px rgba(20,184,166,0.25);
  border: 1px solid rgba(20,184,166,0.2);
  color: #ecfdf5;
  backdrop-filter: blur(20px);
  animation: fadeInHero 1.2s ease;
}
.hero h1 {
  font-size: 2.5rem;
  color: #5eead4;
  text-shadow: 0 0 20px rgba(20,184,166,0.6);
  margin-bottom: 10px;
}
.hero p {
  color: #ccfbf1;
  font-size: 1.1rem;
  margin-bottom: 8px;
}

/* Feature Cards */
.feature-card {
  background: rgba(0,40,37,0.7);
  border-radius: 16px;
  padding: 24px;
  margin: 10px 0;
  text-align: center;
  border: 1px solid rgba(94,234,212,0.15);
  box-shadow: 0 0 25px rgba(0,0,0,0.4);
  backdrop-filter: blur(12px);
  transition: all 0.3s ease;
}
.feature-card:hover {
  transform: translateY(-6px);
  border: 1px solid rgba(94,234,212,0.5);
  box-shadow: 0 0 25px rgba(20,184,166,0.4);
}

/* Sidebar Buttons */
div.stButton > button {
  background: linear-gradient(135deg, #14b8a6, #0d9488);
  color: white;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.5rem 1rem;
  border: none;
  transition: all 0.3s ease;
}
div.stButton > button:hover {
  background: linear-gradient(135deg, #34d399, #059669);
  transform: translateY(-2px);
}

/* Footer */
.footer {
  text-align: center;
  margin-top: 40px;
  color: #94f5df;
  font-size: 0.9em;
  text-shadow: 0 0 12px rgba(94,234,212,0.4);
}

/* Data Table */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(20,184,166,0.3);
  background: rgba(0,0,0,0.3);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb {
  background: rgba(45,212,191,0.4);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(45,212,191,0.8);
}

/* Animation */
@keyframes fadeInHero {
  from { opacity: 0; transform: translateY(30px); }
  to { opacity: 1; transform: translateY(0); }
}

/* =============================== */
/* ⚡ POWER-ON ANIMATION SYSTEM */
/* =============================== */

/* Apply fade-in only to main content wrapper */
section.main {
  opacity: 0;
  animation: fadeInBody 1.2s ease forwards;
  animation-delay: 0.3s;
}

/* Hero section smooth entry */
.hero {
  opacity: 0;
  transform: translateY(40px);
  animation: fadeInHero 1.4s ease forwards;
  animation-delay: 1.0s;
}

/* Feature cards slide-up sequence */
.feature-card {
  opacity: 0;
  transform: translateY(20px);
  animation: fadeInCards 0.8s ease forwards;
  animation-delay: 1.5s;
}

/* Tabs fade + glow in */
[data-baseweb="tab-list"] {
  opacity: 0;
  animation: fadeInTabs 1.2s ease forwards;
  animation-delay: 0.8s;
}

/* Sidebar fade glow */
[data-testid="stSidebar"] {
  opacity: 0;
  animation: fadeInSidebar 1.6s ease forwards;
  animation-delay: 0.3s;
}

/* Animations */
@keyframes fadeInBody {
  from { opacity: 0; filter: blur(8px); }
  to { opacity: 1; filter: blur(0px); }
}

@keyframes fadeInHero {
  from { opacity: 0; transform: translateY(40px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInTabs {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInSidebar {
  from { opacity: 0; transform: translateX(-20px); }
  to { opacity: 1; transform: translateX(0); opacity: 1; }
}

@keyframes fadeInCards {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}


/* ================================ */
/* ✨ Hover Glow for Interactive UI */
/* ================================ */

/* Glow effect for dropdown (selectbox) */
div[data-baseweb="select"] > div:hover {
    box-shadow: 0 0 8px rgba(45, 212, 191, 0.45);
    border-color: #34d399 !important;
    transform: scale(1.02);
    transition: all 0.2s ease-in-out;
}

/* Glow on hover for radio buttons and checkboxes */
div[role="radiogroup"] label:hover,
div[role="checkbox"] label:hover {
    background: rgba(45, 212, 191, 0.12);
    box-shadow: 0 0 10px rgba(94, 234, 212, 0.25);
    border-radius: 8px;
    transform: translateX(2px);
}

/* Soft glow for active (selected) inputs */
input[type="radio"]:checked + div,
input[type="checkbox"]:checked + div {
    box-shadow: 0 0 10px rgba(94, 234, 212, 0.3);
}

/* Smooth transition for all interactive sidebar elements */
[data-testid="stSidebar"] * {
    transition: all 0.25s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # =========================
    # 🌐 LexiAI Sidebar Header
    # =========================
    st.markdown("""
        <div style='text-align:center; padding:18px 0;'>
            <img src='https://cdn-icons-png.flaticon.com/512/4712/4712103.png' width='70' style='filter: drop-shadow(0 0 8px rgba(94,234,212,0.7));'>
            <h2 style='color:#5eead4; margin-top:10px; font-weight:700; letter-spacing:0.5px;'>LexiAI Control Panel</h2>
            <p style='font-size:13px; color:#a7f3d0; margin-top:-8px; font-weight:400;'>Language Meets Intelligence</p>
        </div>
        <div style='height:1px; background:linear-gradient(90deg, transparent, rgba(94,234,212,0.5), transparent); margin:10px 0 18px;'></div>
    """, unsafe_allow_html=True)

    # 🌍 Language Selector
    st.markdown("""
        <h4 style='color:#5eead4; font-weight:600; margin-bottom:6px;'>🌍 Language Settings</h4>
        <p style='font-size:12.5px; color:#94a3b8; margin-top:-4px;'>Choose your preferred NLP analysis language</p>
    """, unsafe_allow_html=True)
    language = st.selectbox(
        "Choose Analysis Language",
        list(LANG_MAP.keys()),
        help="Select the language for NLP analysis and processing."
    )

    # 🎙 Input Preferences
    st.markdown("""
        <h4 style='color:#5eead4; font-weight:600; margin-top:16px; margin-bottom:6px;'>🎙 Input Settings</h4>
        <p style='font-size:12.5px; color:#94a3b8; margin-top:-4px;'>Select how you want to provide input</p>
    """, unsafe_allow_html=True)
    input_mode = st.radio(
        "Input Mode",
        ["Text", "Microphone"],
        index=0,
        help="Enter text manually or record your voice for transcription."
    )

    # 🧩 Display Settings
    st.markdown("""
        <h4 style='color:#5eead4; font-weight:600; margin-top:16px; margin-bottom:6px;'>🧩 Display Options</h4>
        <p style='font-size:12.5px; color:#94a3b8; margin-top:-4px;'>Control how your analysis results are shown</p>
    """, unsafe_allow_html=True)
    show_table = st.checkbox("📊 Show POS Table", True)
    show_html = st.checkbox("💡 Show Colored Tags", True)
    show_hindi_tags = st.checkbox(
        "🇮🇳 Show Hindi POS Labels",
        value=True if language == "Hindi" else False,
        help="Display Hindi translations for POS tags (Hindi only)."
    )

    # ⚙ Divider
    st.markdown("""
        <div style='height:1px; background:linear-gradient(90deg, transparent, rgba(94,234,212,0.4), transparent); margin:16px 0;'></div>
    """, unsafe_allow_html=True)

    # 🧠 About Section
    st.markdown("""
        <h4 style='color:#5eead4; font-weight:600; margin-bottom:6px;'>🧠 About LexiAI</h4>
    """, unsafe_allow_html=True)
    with st.expander("ℹ️ Learn More", expanded=False):
        st.markdown("""
        **LexiAI** is a multilingual Natural Language Processing platform  
        designed to analyze, correct, and summarize text using AI.  

        🔹 Built with: **Streamlit, Flask, Stanza, and SpaCy**  
        🔹 Developed by: *Saniya Firdos M Nadaf*  
        """, unsafe_allow_html=True)
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)

    # 🚪 Logout button now inside sidebar
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    logout = st.button("🚪 Logout", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if logout:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.markdown(
            "<meta http-equiv='refresh' content='0; url=http://localhost:5000/#login'>",
            unsafe_allow_html=True
        )
        st.stop()
        # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    # 📜 My Analysis History
    st.markdown("""
        <h4 style='color:#5eead4; font-weight:600; margin-bottom:6px;'>📜 My Analysis History</h4>
        <p style='font-size:12.5px; color:#94a3b8; margin-top:-4px;'>View your saved NLP analyses</p>
    """, unsafe_allow_html=True)

    user_email = st.query_params.get("email", [""])[0] or "demo@example.com"

    if "view_history_item" not in st.session_state:
        st.session_state.view_history_item = None

    if user_email:
        try:
            resp = requests.get(f"http://localhost:5000/get_history/{user_email}")
            if resp.status_code == 200:
                history = resp.json()
                if len(history) == 0:
                    st.info("No saved analyses yet.")
                else:
                    with st.container():
                        st.markdown("""
                            <div style="max-height: 300px; overflow-y:auto; border-radius:10px;
                            background: rgba(0,30,30,0.3); padding:8px; box-shadow:inset 0 0 10px rgba(94,234,212,0.1);">
                        """, unsafe_allow_html=True)
                        for i, record in enumerate(reversed(history)):
                            feature = record.get('feature', 'N/A')
                            text_preview = record.get('text', '')[:45] + "..."
                            if st.button(f"📘 {feature} – {text_preview}", key=f"history_{i}", use_container_width=True):
                                st.session_state.view_history_item = record
                                st.session_state.page = "history"
                                st.experimental_rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Could not load analysis history.")
        except Exception:
            st.error("⚠️ Error loading history.")
    # ❤️ Footer
    st.markdown("""
        <div class='sidebar-footer' style='text-align:center; padding-top:10px;'>
            <p style='font-size:12px; color:#9ca3af; margin-bottom:2px;'>Made with <span style="color:#10b981;">❤</span> by <b style="color:#5eead4;">Saniya</b></p>
            <p style='font-size:11px; color:#94f1df;'>LexiAI © 2025</p>
        </div>
    """, unsafe_allow_html=True)


# 🌙 Dynamic sidebar theme styling (dark / light)
st.markdown(f"""
    <style>
        /* Textarea Styling */
        textarea {{
            color: {'#ffffff' if dark_mode else '#000000'} !important;
            background-color: {'#1e1e1e' if dark_mode else '#ffffff'} !important;
        }}

        /* Sidebar general text */
        section[data-testid="stSidebar"] {{
            color: {'#f0f0f0' if dark_mode else '#222222'} !important;
            background-color: {'#111827' if dark_mode else '#f9f9f9'} !important;
        }}

        /* Sidebar labels and headers */
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {{
            color: {'#f9fafb' if dark_mode else '#222222'} !important;
        }}

        /* Sidebar selectboxes, checkboxes, and radio buttons */
        div[data-testid="stSidebar"] div[role="radiogroup"] label, 
        div[data-testid="stSidebar"] div[role="checkbox"] label {{
            color: {'#e5e7eb' if dark_mode else '#222222'} !important;
        }}

        /* Sidebar section dividers */
        hr {{
            border-color: {'#444' if dark_mode else '#ddd'} !important;
        }}
    </style>
""", unsafe_allow_html=True)


# Custom CSS for Full Page Theme Support
css = f"""
    <style>
        body {{ background-color: {'#121212' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; }}
        .main {{ background-color: {'#1e1e1e' if dark_mode else '#f9f9f9'}; }}
        .hero {{ background: {'linear-gradient(135deg, #00695c 0%, #004d40 50%, #00251a 100%)' if dark_mode else 'linear-gradient(135deg, #1976d2 0%, #0d47a1 50%, #1565c0 100%)'}; color: white; padding: 40px; border-radius: 12px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .hero h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 600; }}
        .hero p {{ font-size: 1.1em; margin-bottom: 15px; }}
        .feature-card {{ background: {'#424242' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; padding: 20px; border-radius: 10px; margin: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid {'#666' if dark_mode else '#ddd'}; transition: transform 0.2s; }}
        .feature-card:hover {{ transform: translateY(-5px); }}
        .tab-content {{ padding: 20px; background: {'#2c2c2c' if dark_mode else '#ffffff'}; color: {'#ffffff' if dark_mode else '#333333'}; border-radius: 10px; margin-top: 10px; }}
        .footer {{ text-align: center; margin-top: 40px; color: {'#cccccc' if dark_mode else '#777'}; font-size: 0.9em; }}
        .stButton button {{ border-radius: 8px; font-weight: 500; transition: background 0.3s; }}
        .stButton button:hover {{ background-color: #1976d2 !important; color: white !important; }}
        .stTextArea textarea, .stSelectbox select, .stRadio label {{ color: {'#ffffff' if dark_mode else '#333333'} !important; }}
        .stDataFrame, .stTable {{ background: {'#2c2c2c' if dark_mode else '#ffffff'} !important; color: {'#ffffff' if dark_mode else '#333333'} !important; }}
    </style>
"""

st.markdown(css, unsafe_allow_html=True)  # Apply the CSS

# ======================================
# 🎯 PAGE NAVIGATION CONTROLLER
# ======================================
if st.session_state.page == "home" and not st.session_state.view_history_item:
    # ---- Your existing tab structure goes here ----
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home",
        "🏷 POS Tagger",
        "📝 Grammar Check",
        "💬 Sentiment Analysis",
        "🧠 Entity Recognition",
        "☁ Word Cloud & Frequency",
        "📘 Text Summarizer"
    ])

    # ==========================================
    # Tab 1: Home (Updated)
    # ==========================================
    with tab1:
        st.markdown(f"""
        <div class="hero">
            <h1>💬 LexiAI — Multilingual NLP Intelligence Platform</h1>
            <p>Empowering language understanding through next-generation AI.</p>
            <p>Analyze, visualize, correct, summarize, and speak — all in one intelligent platform.</p>
        </div>
        """, unsafe_allow_html=True)

        # 🌟 Glassmorphic Key Features (Static Icons + Header Animation + New Feature)
        st.markdown("""
        <style>
            /* === Glassmorphic + Slide-up Animation === */
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 22px;
                margin-top: 35px;
            }

            .feature-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 18px;
                padding: 32px 22px;
                text-align: center;
                border: 1px solid rgba(94, 234, 212, 0.25);
                box-shadow: 0 0 25px rgba(0, 0, 0, 0.35), inset 0 0 10px rgba(94,234,212,0.08);
                backdrop-filter: blur(15px);
                transition: all 0.35s ease;
                opacity: 0;
                transform: translateY(25px);
                animation: fadeInCard 0.9s ease forwards;
            }

            /* Animate cards one by one */
            .feature-card:nth-child(1) { animation-delay: 0.2s; }
            .feature-card:nth-child(2) { animation-delay: 0.35s; }
            .feature-card:nth-child(3) { animation-delay: 0.5s; }
            .feature-card:nth-child(4) { animation-delay: 0.65s; }
            .feature-card:nth-child(5) { animation-delay: 0.8s; }
            .feature-card:nth-child(6) { animation-delay: 0.95s; }
            .feature-card:nth-child(7) { animation-delay: 1.1s; }
            .feature-card:nth-child(8) { animation-delay: 1.25s; }
            .feature-card:nth-child(9) { animation-delay: 1.4s; }

            @keyframes fadeInCard {
                from { opacity: 0; transform: translateY(25px) scale(0.97); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }

            .feature-card:hover {
                transform: translateY(-6px) scale(1.04);
                border: 1px solid rgba(94, 234, 212, 0.6);
                box-shadow: 0 0 35px rgba(94, 234, 212, 0.4), inset 0 0 10px rgba(94,234,212,0.1);
            }

            .feature-icon {
                font-size: 2.2rem;
                color: #5eead4;
                text-shadow: 0 0 10px rgba(94,234,212,0.6);
                display: block;
                margin-bottom: 12px;
            }

            .feature-title {
                color: #e0f2f1;
                font-weight: 700;
                font-size: 1.2rem;
                margin-bottom: 8px;
                letter-spacing: 0.3px;
            }

            .feature-desc {
                color: #cbd5e1;
                font-size: 0.95rem;
                line-height: 1.6;
            }

            /* Header animation */
            .features-heading {
                text-align: center;
                color: #5eead4;
                margin-top: 30px;
                font-weight: 700;
                letter-spacing: 0.5px;
                font-size: 1.8rem;
                opacity: 0;
                transform: translateY(30px);
                animation: slideUp 0.9s ease forwards;
            }

            .features-subtext {
                text-align: center;
                color: #a7f3d0;
                font-size: 14px;
                margin-bottom: 20px;
                opacity: 0;
                transform: translateY(20px);
                animation: slideUp 1s ease forwards;
                animation-delay: 0.3s;
            }

            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>

        <h2 class="features-heading"> Explore Key Features</h2>
        <p class="features-subtext">Discover the core capabilities that make LexiAI your all-in-one NLP intelligence platform.</p>

        <div class="features-grid">
            <div class="feature-card">
                <span class="feature-icon">🏷️</span>
                <h4 class="feature-title">POS Tagger</h4>
                <p class="feature-desc">Analyze grammatical structures across English and Indian languages using <b>Stanza</b>.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">📝</span>
                <h4 class="feature-title">Grammar Correction</h4>
                <p class="feature-desc">Fix English grammar errors instantly with <b>AI-powered LanguageTool</b> integration.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">💬</span>
                <h4 class="feature-title">Sentiment Analysis</h4>
                <p class="feature-desc">Detect emotional tone — Positive, Negative, or Neutral — using <b>TextBlob</b>.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🧠</span>
                <h4 class="feature-title">Entity Recognition</h4>
                <p class="feature-desc">Automatically identify <b>people, organizations, and places</b> within your text.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">☁️</span>
                <h4 class="feature-title">Word Cloud</h4>
                <p class="feature-desc">Visualize frequent words beautifully and intuitively through <b>WordCloud</b> generation.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">📘</span>
                <h4 class="feature-title">Text Summarizer</h4>
                <p class="feature-desc">Condense long paragraphs into concise summaries using <b>Transformer models</b>.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🎙️</span>
                <h4 class="feature-title">Speech Recognition</h4>
                <p class="feature-desc">Convert voice to text seamlessly using <b>Google Speech API</b>.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🌐</span>
                <h4 class="feature-title">Multilingual Support</h4>
                <p class="feature-desc">Supports <b>English, Hindi, Marathi, Tamil, Telugu, and Urdu</b> seamlessly.</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <h4 class="feature-title">Real-time Analytics</h4>
                <p class="feature-desc">Visualize linguistic insights, trends, and frequency charts instantly for each analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='
                background: linear-gradient(145deg, rgba(13, 148, 136, 0.15), rgba(16, 185, 129, 0.08));
                border: 1px solid rgba(94, 234, 212, 0.25);
                border-radius: 16px;
                padding: 30px 40px;
                margin-top: 30px;
                box-shadow: 0 4px 25px rgba(0,0,0,0.3);
                font-family: "Poppins", sans-serif;
            '>
                <h2 style='color:#5eead4; text-align:center; font-weight:700; letter-spacing:0.5px; margin-bottom:10px;'>
                    Quick Start Guide
                </h2>
                <p style='color:#cbd5e1; text-align:center; font-size:14px; margin-bottom:25px;'>
                    Get started with LexiAI in just a few simple steps.
                </p>
                <hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(94,234,212,0.6), transparent);'>
                <ol style='color:#e5e5e5; font-size:15px; line-height:1.8; margin-top:20px;'>
                    <li><b style='color:#5eead4;'>Select Language</b> from the sidebar to choose your preferred NLP analysis language.</li>
                    <li><b style='color:#5eead4;'>Provide Input</b> using Text or Microphone mode for speech-to-text analysis.</li>
                    <li><b style='color:#5eead4;'>Navigate Features</b> through the top navigation bar:</li>
                    <ul style='list-style-type:square; margin-left:25px;'>
                        <li>🧩 <b>POS Tagger</b> — Analyze parts of speech and grammar structure.</li>
                        <li>📝 <b>Grammar Check</b> — Instantly correct English grammar mistakes.</li>
                        <li>💬 <b>Sentiment Analysis</b> — Detect emotional tone and polarity.</li>
                        <li>🧠 <b>Entity Recognition</b> — Identify people, organizations, and places.</li>
                        <li>☁️ <b>Word Cloud</b> — Visualize frequent words beautifully.</li>
                        <li>📘 <b>Text Summarizer</b> — Condense long paragraphs using AI models.</li>
                    </ul>
                    <li><b style='color:#5eead4;'>Save & Export</b> your NLP results as CSV or text files for future reference.</li>
                </ol>
                <p style='color:#9ca3af; text-align:center; margin-top:25px; font-style:italic;'>
                    💡 Tip: You can view your complete analysis history in the sidebar anytime!
                </p>
            </div>
        """, unsafe_allow_html=True)
    # ==========================================
    # Tab 2: POS Tagger
    # ==========================================
    with tab2:
        st.header("🏷 Part-of-Speech Tagger")

        # --- Input mode: text or voice ---
        if input_mode == "Text":
            input_text = st.text_area(
                "Enter or paste text:",
                height=150,
                placeholder="Type your text here...",
                key="pos_text_input"
            )
        else:
            st.write("🎤 Click to Record")
            if st.button("🎙 Start Recording", key="pos_record_button"):
                spoken = transcribe_from_mic(language)
                st.success(f"✅ You said: {spoken}")
                st.session_state["voice_text"] = spoken

            input_text = st.session_state.get("voice_text", "")
            st.text_area(
                "Transcribed Text:",
                value=input_text,
                height=150,
                disabled=True,
                key="pos_voice_input"
            )

        # --- Analyze POS button (now visible for both modes) ---
        if st.button("🔍 Analyze POS", key="pos_analyze") and input_text.strip():
            nlp = init_stanza(LANG_MAP[language])

            if nlp is not None:
                progress = st.progress(0)
                with st.spinner("Processing..."):
                    for i in range(100):
                        progress.progress(i + 1)
                    rows = stanza_pos(nlp, input_text)

                if not rows:
                    st.error("No tokens detected. Check input.")
                else:
                    df = pd.DataFrame(rows, columns=["Token", "UPOS", "XPOS"])

                    if show_hindi_tags and language == "Hindi":
                        df["UPOS (Hindi)"] = df["UPOS"].map(lambda t: POS_TAGS_HINDI.get(t, t))

                    col1, col2 = st.columns(2)
                    with col1:
                        if show_table:
                            st.dataframe(df, use_container_width=True)
                    with col2:
                        if show_html:
                            st.markdown("*Visual POS Tags:*")
                            st.markdown(
                                render_tagged_html(rows, show_hindi_tags, language),
                                unsafe_allow_html=True
                            )

                    # --- Statistics ---
                    st.subheader("📊 Statistics")
                    pos_counts = df["UPOS"].value_counts()
                    total = pos_counts.sum()
                    percentage = (pos_counts / total * 100).round(2)
                    stats_df = pd.DataFrame({
                        "Tag": pos_counts.index,
                        "Count": pos_counts.values,
                        "%": percentage.values
                    })
                    st.dataframe(stats_df, use_container_width=True)

                    # --- Charts ---
                    import plotly.express as px
                    stats_df_sorted = stats_df.sort_values("Count", ascending=False)
                    chart_tab1, chart_tab2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                    with chart_tab1:
                        fig_bar = px.bar(
                            stats_df_sorted, x="Count", y="Tag", orientation="h",
                            text=stats_df_sorted["%"].apply(lambda x: f"{x:.1f}%"),
                            color="Count", color_continuous_scale="Tealgrn" if dark_mode else "Blues",
                            title="POS Tag Frequency Distribution"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with chart_tab2:
                        fig_pie = px.pie(
                            stats_df_sorted, names="Tag", values="Count",
                            title="POS Tag Proportion (%)", hole=0.4
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # --- Save analysis result to Flask backend ---
                    try:
                        save_result_to_db(user_email, input_text, "POS Tagging", df.to_json(orient="records"))
                    except Exception as e:
                        st.warning("Could not save POS result (non-critical).")

                    # --- Download button ---
                    st.download_button(
                        "⬇ Export CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "pos_tags.csv",
                        "text/csv"
                    )


    # ---------- TAB 3: Grammar Check ----------
    with tab3:
        st.header("📝 Grammar Correction")

        if language != "English":
            st.warning("Grammar check is only available for English.")
        else:
            input_text = st.text_area(
                "Enter text for grammar check:",
                height=150,
                placeholder="Paste English text here..."
            )

            if st.button("🔍 Check Grammar", key="grammar_check") and input_text.strip():
                progress = st.progress(0)
                with st.spinner("Analyzing..."):
                    for i in range(100):
                        progress.progress(i + 1)
                    corrected, errors = check_grammar(input_text, language)

                st.subheader("Corrected Text")
                st.write(corrected)

                st.subheader("Issues Found")
                if len(errors) > 0:
                    st.dataframe(errors, use_container_width=True)
                else:
                    st.success("No errors detected!")


    # ---------- TAB 4: Sentiment Analysis ----------
    with tab4:
        st.markdown("## 💬 Sentiment Analysis")
        st.markdown("Analyze the emotional tone of your text — Positive 😊, Negative 😠, or Neutral 😐.")

        # --- Input Box ---
        input_text = st.text_area(
            "Enter text for sentiment analysis:",
            height=150,
            placeholder="Type or paste text here to analyze sentiment..."
        )

        # --- Analyze Button ---
        if st.button("🔍 Analyze Sentiment", key="sentiment_analyze"):
            if input_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    blob = TextBlob(input_text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity

                    # Determine sentiment type and color
                    if polarity > 0.1:
                        sentiment_label = "😊 Positive"
                        st.success("😊 Positive sentiment detected!")
                        color = "#10b981"  # green
                    elif polarity < -0.1:
                        sentiment_label = "😠 Negative"
                        st.error("😠 Negative sentiment detected.")
                        color = "#ef4444"  # red
                    else:
                        sentiment_label = "😐 Neutral"
                        st.info("😐 Balanced or factual statement.")
                        color = "#facc15"  # yellow

                    # --- Sentiment Gauge Visualization ---
                    st.markdown("### 📊 Sentiment Polarity Gauge")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=polarity * 100,
                        title={"text": "Sentiment Score (Polarity)", "font": {"size": 18}},
                        number={"suffix": " %", "font": {"color": color, "size": 22}},
                        gauge={
                            "axis": {"range": [-100, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [-100, -30], "color": "rgba(239,68,68,0.4)"},  # red zone
                                {"range": [-30, 30], "color": "rgba(250,204,21,0.4)"},   # yellow zone
                                {"range": [30, 100], "color": "rgba(16,185,129,0.4)"}    # green zone
                            ],
                        },
                        delta={
                            "reference": 0,
                            "increasing": {"color": "#10b981"},
                            "decreasing": {"color": "#ef4444"}
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Display numerical values ---
                    st.markdown(f"""
                        **Polarity:** `{polarity:.2f}`  
                        **Subjectivity:** `{subjectivity:.2f}`
                    """)

                    # --- Save Result to Backend ---
                    try:
                        save_result_to_db(user_email, input_text, "Sentiment Analysis", sentiment_label)
                    except Exception:
                        st.warning("⚠️ Could not save sentiment result (non-critical).")
            else:
                st.warning("Please enter some text to analyze.")

    # ================================
    # Tab 5: Entity Recognition (Fixed)
    # ================================
    with tab5:
        st.header("🧠 Named Entity Recognition (NER)")
        st.write("Automatically identify and categorize named entities like names, organizations, and locations.")

        input_text = st.text_area(
            "Enter text for entity recognition:",
            height=150,
            placeholder="Type or paste English text here..."
        )

        # ---- Main Extraction Logic ----
        if st.button("🔍 Extract Entities"):
            try:
                # Load SpaCy model
                nlp = spacy.load("en_core_web_trf")
                doc = nlp(input_text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]

                if entities:
                    # Show entity table
                    df_ner = pd.DataFrame(entities, columns=["Entity", "Label"])
                    st.dataframe(df_ner, use_container_width=True)

                    # 🧭 Entity Legend Bar
                    st.markdown("""
                    <div style='margin: 20px 0 10px; text-align:center;'>
                    <span style='background:#c7d2fe; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>🧍 PERSON</span>
                    <span style='background:#99f6e4; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>🏢 ORG</span>
                    <span style='background:#fde68a; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>🌍 GPE</span>
                    <span style='background:#bbf7d0; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>🛒 PRODUCT</span>
                    <span style='background:#bae6fd; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>📅 DATE</span>
                    <span style='background:#fecaca; color:#000; padding:6px 12px; border-radius:6px; margin:3px; display:inline-block;'>💰 MONEY</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # 🧠 Section Title
                    st.markdown("### 🔍 Highlighted Entities in Text")

                    # ✨ Hover Glow Effect
                    st.markdown("""
                    <style>
                    mark.entity:hover {
                    box-shadow: 0 0 12px rgba(94,234,212,0.8);
                    transform: scale(1.05);
                    transition: all 0.2s ease-in-out;
                    border-radius: 4px;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # 🧩 Show Highlighted Text
                    html = spacy.displacy.render(doc, style="ent", jupyter=False)
                    st.markdown(html, unsafe_allow_html=True)

                    # ⬇️ Export Option
                    st.download_button(
                        "⬇ Export Entities",
                        df_ner.to_csv(index=False).encode("utf-8"),
                        "entities.csv",
                        "text/csv"
                    )

                else:
                    st.info("No named entities found in the text.")

            except Exception as e:
                st.error(f"Error: {e}")


    # ================================
    # Tab 6: Word Cloud & Frequency
    # ================================
    with tab6:
        st.header("☁ Word Cloud & Frequency")
        st.write("Visualize the most frequent words in your text.")

        input_text = st.text_area(
            "Enter text for word cloud:",
            height=150,
            placeholder="Paste text here..."
        )

        if st.button("☁ Generate Word Cloud"):
            if input_text.strip():
                words = input_text.split()
                word_freq = Counter(words)

                # Display top 15 words
                freq_df = pd.DataFrame(word_freq.most_common(15), columns=["Word", "Frequency"])
                st.dataframe(freq_df, use_container_width=True)

                # Word Cloud generation
                wc = WordCloud(
                    width=900,
                    height=450,
                    background_color="#0b1c1a",  # dark teal background
                    colormap="winter",           # blue-green color map
                    prefer_horizontal=0.9,
                    contour_color="#5eead4",     # teal border
                    contour_width=1
                ).generate(input_text)

                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # Download word frequency CSV
                st.download_button(
                    "⬇ Export Word Frequency",
                    freq_df.to_csv(index=False).encode("utf-8"),
                    "word_frequency.csv",
                    "text/csv"
                )
            else:
                st.warning("Please enter some text first.")


    # ================================
    # Tab 7: Text Summarization
    # ================================
    with tab7:
        st.markdown("## 🧠 Text Summarization")
        st.markdown("Generate concise summaries from large text passages.")

        input_text = st.text_area("Enter text to summarize:", height=200)

        if st.button("Summarize Text", key="summarize"):
            if input_text.strip():
                with st.spinner("Summarizing..."):
                    try:
                        # ✅ Use Sumy instead of Gensim
                        from sumy.parsers.plaintext import PlaintextParser
                        from sumy.nlp.tokenizers import Tokenizer
                        from sumy.summarizers.lsa import LsaSummarizer

                        parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
                        summarizer = LsaSummarizer()
                        summary = summarizer(parser.document, sentences_count=3)
                        summarized_text = " ".join(str(sentence) for sentence in summary)

                        if not summarized_text.strip():
                            st.warning("Input text too short for summarization.")
                        else:
                            st.subheader("📝 Summary:")
                            st.success(summarized_text)

                            # --- Save to analysis history ---
                            try:
                                save_result_to_db(user_email, input_text, "Summarization", summarized_text)
                            except Exception:
                                st.warning("Could not save summary result (non-critical).")

                            # --- Download summary button ---
                            st.download_button(
                                "⬇ Download Summary",
                                summarized_text.encode("utf-8"),
                                "summary.txt",
                                "text/plain"
                            )
                    except ModuleNotFoundError:
                        st.error("⚠ Sumy module not found. Please install it using: pip install sumy")
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
            else:
                st.warning("Please enter some text before summarizing.")

# ======================================
# 🪄 FULL PAGE HISTORY VIEW (FIXED)
# ======================================
if st.session_state.page == "history" and st.session_state.view_history_item:
    record = st.session_state.view_history_item
    st.markdown("---")
    st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, rgba(0,64,59,0.9), rgba(0,90,78,0.9));
            border-radius: 20px;
            padding: 40px;
            color: #e0fdf7;
            box-shadow: 0 0 25px rgba(20,184,166,0.25);
            margin-top: 20px;
        ">
            <h2 style="color:#5eead4; font-weight:700;">🧩 {record['feature']}</h2>
            <p style="color:#a7f3d0; font-size:14px;">📅 <b>{record.get('timestamp', 'Recently')}</b></p>
            <hr style="border:none; height:1px; background:rgba(94,234,212,0.4); margin:10px 0;">
            <h4 style="color:#5eead4;">🔹 Input Text</h4>
            <p style="background:rgba(0,40,38,0.5); padding:10px; border-radius:8px;">{record['text']}</p>
            <h4 style="color:#5eead4; margin-top:15px;">🔹 Result</h4>
            <p style="background:rgba(0,40,38,0.5); padding:10px; border-radius:8px;">{record['result']}</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "home"
        st.session_state.view_history_item = None
        st.rerun()




# ===========================================
# 💅 Sidebar & Main Page Theming Fix
# ===========================================
st.markdown(f"""
    <style>
        /* Sidebar background & text */
        [data-testid="stSidebar"] {{
            background-color: {'#1e1e2f' if dark_mode else '#f9f9f9'};
            color: {'#f1f1f1' if dark_mode else '#111111'};
            transition: all 0.3s ease-in-out;
        }}

        /* Sidebar headers and section titles */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h4 {{
            color: {'#f8f9fa' if dark_mode else '#111111'} !important;
        }}

        /* Sidebar labels */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
            color: {'#eaeaea' if dark_mode else '#333333'} !important;
        }}

        /* Sidebar selectboxes and checkboxes */
        div[data-baseweb="select"] > div {{
            background-color: {'#2b2b3d' if dark_mode else '#ffffff'} !important;
            color: {'#f8f9fa' if dark_mode else '#000000'} !important;
            border-radius: 8px;
            border: 1px solid {'#444' if dark_mode else '#ccc'} !important;
        }}

        div[role="radiogroup"] label p,
        div[role="checkbox"] label p {{
            color: {'#fafafa' if dark_mode else '#222222'} !important;
            font-size: 15px;
            font-weight: 500;
        }}

        /* Expander background */
        div[data-testid="stExpander"] {{
            background-color: {'#2b2b3d' if dark_mode else '#ffffff'} !important;
            border-radius: 10px;
            border: 1px solid {'#333' if dark_mode else '#ddd'} !important;
        }}

        /* Tabs text color fix */
        button[data-baseweb="tab"] {{
            color: {'#f8f9fa' if dark_mode else '#111111'} !important;
            font-weight: 600;
            font-size: 15px;
        }}
        button[data-baseweb="tab"]:hover {{
            background-color: {'#333c52' if dark_mode else '#e0e0e0'} !important;
        }}

        /* Main text area styling */
        textarea {{
            color: {'#ffffff' if dark_mode else '#000000'} !important; 
            background-color: {'#1e1e1e' if dark_mode else '#ffffff'} !important;
        }}

        /* Streamlit metric and dataframe color fix */
        div[data-testid="stDataFrame"] {{
            color: {'#e0e0e0' if dark_mode else '#000000'} !important;
        }}

    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        /* Make all tab text white */
        button[data-baseweb="tab"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Highlight the active tab with a subtle underline */
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #4ea8de !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }

        /* Optional: change hover color */
        button[data-baseweb="tab"]:hover p {
            color: #90caf9 !important;
        }
    </style>
""", unsafe_allow_html=True)




# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p><b>LexiAI — Multilingual NLP Intelligence Platform</b> — Explore, Analyze, and Understand Language with AI.</p>
        <p>Supports multilingual NLP tasks including POS tagging, grammar correction, sentiment analysis, entity recognition, and more.</p>
<p>💡 Developed with passion by <b>Saniya Nadaf</b> | LexiAI © 2025</p>
    </div>
""", unsafe_allow_html=True)