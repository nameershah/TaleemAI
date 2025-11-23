import streamlit as st
import os
from PyPDF2 import PdfReader
from gtts import gTTS
import io

# --- 0. CONFIGURATION ---
st.set_page_config(
    page_title="TaleemAI - Agentic Tutor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. SECRETS SETUP (With Debug Prints) ---
# Try loading from Environment (HF) or Streamlit Secrets (Local)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# Initialize Clients
groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)

genai = None
if GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF Read Error: {e}")
        return ""

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp
    except Exception as e:
        print(f"Audio Error: {e}")
        return None

# --- 3. LLM FUNCTIONS (With Detailed Error Reporting) ---
def call_llm_groq(prompt, context=""):
    if not groq_client: return "⚠️ GROQ NOT READY: Check API Key."
    full_prompt = prompt
    if context: full_prompt = f"Context: {context[:6000]}\n\nTask: {prompt}"
    
    st.toast("⚡ Routing to GROQ", icon="🟢")
    try:
        out = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return out.choices[0].message.content
    except Exception as e:
        return f"❌ GROQ ERROR: {e}"

def call_llm_gemini(prompt, context=""):
    if not genai: return "⚠️ GEMINI NOT READY: Check API Key."
    full_prompt = prompt
    if context: full_prompt = f"Context: {context[:40000]}\n\nQuestion: {prompt}"
    
    st.toast("🧠 Routing to GEMINI", icon="🟣")
    
    # Try models one by one
    models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
    last_error = ""
    
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            continue 
            
    # If we get here, all models failed. Return the REAL error.
    return f"❌ GEMINI ERROR: {last_error}"

# --- 4. ROUTER ---
def agentic_router(user_input, context_text=""):
    if not genai: return call_llm_groq(user_input, context_text), "GROQ"

    if len(user_input) > 200: return call_llm_gemini(user_input, context_text), "GEMINI"
    
    keywords = ["summarize", "explain", "study", "math", "code", "detailed"]
    if any(k in user_input.lower() for k in keywords):
        return call_llm_gemini(user_input, context_text), "GEMINI"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(f"Classify '{user_input}' as GROQ or GEMINI").text
        if "GEMINI" in res.upper(): return call_llm_gemini(user_input, context_text), "GEMINI"
    except:
        pass
        
    return call_llm_groq(user_input, context_text), "GROQ"

# --- 5. STYLING ---
st.markdown("""
<style>
    .stButton button { background-color: #00FFA3 !important; color: black !important; border-radius: 8px; }
    .badge-groq { background: linear-gradient(135deg, #00FFA3 0%, #00CC82 100%); color: black; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .badge-gemini { background: linear-gradient(135deg, #9D4EDD 0%, #7B2CBF 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .response-box { border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(128, 128, 128, 0.2); background-color: rgba(255, 255, 255, 0.05); }
</style>
""", unsafe_allow_html=True)

# --- 6. MAIN UI ---
st.title("👋 Welcome to TaleemAI")

with st.sidebar:
    st.header("🔧 Diagnostics")
    
    # DEBUG: Check if keys exist (Masked)
    if GOOGLE_API_KEY:
        st.success(f"✅ Google Key Loaded (Ends in ...{GOOGLE_API_KEY[-4:]})")
    else:
        st.error("❌ Google Key Missing!")
        
    if GROQ_API_KEY:
        st.success(f"✅ Groq Key Loaded (Ends in ...{GROQ_API_KEY[-4:]})")
    else:
        st.error("❌ Groq Key Missing!")
        
    st.markdown("---")
    st.header("📁 Materials")
    uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
    pdf_context = extract_text_from_pdf(uploaded_file) if uploaded_file else ""
    if pdf_context: st.info(f"📄 PDF Loaded ({len(pdf_context)} chars)")

user_query = st.text_area("Ask your AI Tutor:", height=120, placeholder="e.g. Summarize this PDF...")

if st.button("🚀 Run Agent"):
    if not user_query:
        st.warning("⚠️ Please type a question.")
    elif not GROQ_API_KEY or not GOOGLE_API_KEY:
        st.error("🚨 API Keys Missing! Check Sidebar Diagnostics.")
    else:
        with st.spinner("🤖 Orchestrating Agents..."):
            response_text, route_used = agentic_router(user_query, pdf_context)
            
            badge = "badge-groq" if route_used == "GROQ" else "badge-gemini"
            st.markdown(f'<div class="{badge}">Routed to {route_used}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="response-box">{response_text}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            with st.spinner("🎤 Generating audio..."):
                audio_file = text_to_speech(response_text[:500])
                if audio_file: st.audio(audio_file, format='audio/mp3')
                else: st.caption("Audio skipped (Rate Limit)")