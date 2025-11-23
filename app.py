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

# --- 1. SECRETS SETUP ---
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

# --- 3. DYNAMIC MODEL FINDER (The Fix) ---
def get_gemini_model_name():
    """
    Dynamically asks Google which models are available 
    to prevent '404 Model Not Found' errors.
    """
    if not genai: return None
    
    try:
        # Ask the API what is valid
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
        
        # 1. Prefer Flash (Fastest/Newest)
        for m in valid_models:
            if "flash" in m and "1.5" in m: return m
        
        # 2. Fallback to Pro
        for m in valid_models:
            if "pro" in m and "1.5" in m: return m
            
        # 3. Last Resort: Take anything valid
        if valid_models: return valid_models[0]
        
    except Exception as e:
        # If listing fails, fall back to the most common standard string
        print(f"Model List Error: {e}")
    
    return "gemini-1.5-flash"

# --- 4. LLM FUNCTIONS ---
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
    
    # Use the dynamic finder
    model_name = get_gemini_model_name()
    
    full_prompt = prompt
    if context: full_prompt = f"Context: {context[:40000]}\n\nQuestion: {prompt}"
    
    st.toast(f"🧠 Routing to GEMINI ({model_name})", icon="🟣")
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ GEMINI ERROR: {e}"

# --- 5. ROUTER ---
def agentic_router(user_input, context_text=""):
    if not genai: return call_llm_groq(user_input, context_text), "GROQ"

    if len(user_input) > 200: return call_llm_gemini(user_input, context_text), "GEMINI"
    
    keywords = ["summarize", "explain", "study", "math", "code", "detailed"]
    if any(k in user_input.lower() for k in keywords):
        return call_llm_gemini(user_input, context_text), "GEMINI"

    # AI Decision
    try:
        model_name = get_gemini_model_name()
        model = genai.GenerativeModel(model_name)
        res = model.generate_content(f"Classify '{user_input}' as GROQ or GEMINI").text
        if "GEMINI" in res.upper(): return call_llm_gemini(user_input, context_text), "GEMINI"
    except:
        pass
        
    return call_llm_groq(user_input, context_text), "GROQ"

# --- 6. STYLING & UI ---
st.markdown("""
<style>
    .stButton button { background-color: #00FFA3 !important; color: black !important; border-radius: 8px; }
    .badge-groq { background: linear-gradient(135deg, #00FFA3 0%, #00CC82 100%); color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .badge-gemini { background: linear-gradient(135deg, #9D4EDD 0%, #7B2CBF 100%); color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .response-box { border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(128, 128, 128, 0.2); background-color: rgba(255, 255, 255, 0.05); }
</style>
""", unsafe_allow_html=True)

st.title("👋 Welcome to TaleemAI")

with st.sidebar:
    st.header("📁 Materials")
    uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
    pdf_context = extract_text_from_pdf(uploaded_file) if uploaded_file else ""
    if pdf_context: st.success(f"✅ Loaded {len(pdf_context)} chars")

user_query = st.text_area("Ask your AI Tutor:", height=120)

if st.button("🚀 Run Agent"):
    if not user_query:
        st.warning("⚠️ Please type a question.")
    elif not GROQ_API_KEY or not GOOGLE_API_KEY:
        st.error("🚨 API Keys Missing!")
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