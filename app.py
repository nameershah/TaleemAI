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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
    """Extracts text from uploaded PDF for RAG context."""
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
    """Converts text to audio using gTTS with Crash Guard."""
    try:
        # Generate audio using Google TTS
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp
    except Exception as e:
        # If it fails (429 or other), return None so the app keeps running
        print(f"Audio failed: {e}") 
        return None

# --- 3. LLM FUNCTIONS ---
def call_llm_groq(prompt, context=""):
    """Calls Groq (Llama 3.1)."""
    if not groq_client:
        return "⚠️ GROQ NOT READY: Check API Key."
    
    full_prompt = prompt
    if context:
        full_prompt = f"Context: {context[:6000]}\n\nTask: {prompt}"
    
    st.toast("⚡ Routing to GROQ (Speed Mode)", icon="🟢")
    
    try:
        out = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return out.choices[0].message.content
    except Exception as e:
        return f"❌ GROQ ERROR: {e}"

def call_llm_gemini(prompt, context=""):
    """Calls Gemini with Fallback Models."""
    if not genai:
        return "⚠️ GEMINI NOT READY: Check API Key."
    
    full_prompt = prompt
    if context:
        # 40k char limit for context
        full_prompt = f"Context: {context[:40000]}\n\nQuestion: {prompt}"
    
    st.toast("🧠 Routing to GEMINI (Precision Mode)", icon="🟣")
    
    # Try different model names in case Google changes API paths
    models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
    
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(full_prompt)
            return response.text
        except:
            continue
            
    return "❌ GEMINI ERROR: Could not connect to any model. Check API Key."

# --- 4. THE ROUTER ---
def agentic_router(user_input, context_text=""):
    if not genai: return call_llm_groq(user_input, context_text), "GROQ"

    # Rules
    if len(user_input) > 200: return call_llm_gemini(user_input, context_text), "GEMINI"
    
    keywords = ["summarize", "explain", "study", "math", "code"]
    if any(k in user_input.lower() for k in keywords):
        return call_llm_gemini(user_input, context_text), "GEMINI"

    # AI Decision
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(f"Classify '{user_input}' as GROQ or GEMINI").text
        if "GEMINI" in res.upper(): return call_llm_gemini(user_input, context_text), "GEMINI"
    except:
        pass
        
    return call_llm_groq(user_input, context_text), "GROQ"

# --- 5. UI STYLES ---
st.markdown("""
<style>
    .stButton button { background-color: #00FFA3 !important; color: black !important; border-radius: 8px; }
    .badge-groq { background: #00FFA3; color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .badge-gemini { background: #9D4EDD; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
    .response-box { border: 1px solid #444; padding: 20px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- 6. MAIN APP ---
st.title("👋 TaleemAI Tutor")

with st.sidebar:
    st.header("Upload Materials")
    f = st.file_uploader("Upload PDF", type=['pdf'])
    pdf_text = extract_text_from_pdf(f) if f else ""
    if pdf_text: st.success(f"Loaded {len(pdf_text)} chars")

q = st.text_area("Ask a question:", height=100)

if st.button("🚀 Run Agent"):
    if not q: st.warning("Type a question!")
    else:
        with st.spinner("Processing..."):
            ans, route = agentic_router(q, pdf_text)
            
            # Show Badge
            badge = "badge-groq" if route == "GROQ" else "badge-gemini"
            st.markdown(f'<div class="{badge}">Routed to {route}</div>', unsafe_allow_html=True)
            
            # Show Answer
            st.markdown(f'<div class="response-box">{ans}</div>', unsafe_allow_html=True)
            
            # Audio (Safe Mode)
            st.markdown("---")
            audio = text_to_speech(ans[:500])
            if audio:
                st.audio(audio, format='audio/mp3')
            else:
                st.info("ℹ️ Audio temporarily unavailable (Rate Limit Protection)")