import streamlit as st
import os
from PyPDF2 import PdfReader
import pyttsx3
import threading

# --- 0. CONFIGURATION ---
st.set_page_config(
    page_title="TaleemAI - Agentic Tutor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. SECRETS SETUP ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Clients safely
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

def text_to_speech(text, filename="output.mp3"):
    """
    Converts text to audio using OFFLINE pyttsx3 engine.
    This avoids the '429 Too Many Requests' error from Google.
    """
    try:
        # Initialize engine
        engine = pyttsx3.init()
        
        # Optimize voice (optional: speed up slightly)
        engine.setProperty('rate', 150) 
        
        # Save to file
        engine.save_to_file(text, filename)
        engine.runAndWait()
        
        return filename
    except Exception as e:
        st.warning(f"Audio generation skipped (Offline TTS Error): {e}")
        return None

# --- 3. LLM FUNCTIONS ---
def call_llm_groq(prompt, context=""):
    """Calls Groq (Llama 3.1) for fast generation."""
    if not groq_client:
        return "⚠️ GROQ NOT READY: Check API Key."
    
    full_prompt = prompt
    if context:
        full_prompt = f"Context: {context[:6000]}\n\nTask: {prompt}"
    
    st.toast("⚡ Routing to GROQ (Llama 3.1 Speed Mode)", icon="🟢")
    
    try:
        out = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return out.choices[0].message.content
    except Exception as e:
        return f"❌ GROQ ERROR (CRASH AVOIDED): {e}"

def call_llm_gemini(prompt, context=""):
    """Calls Gemini with FALLBACK logic for model names."""
    if not genai:
        return "⚠️ GEMINI NOT READY: Check API Key."
    
    full_prompt = prompt
    if context:
        full_prompt = f"Context: {context[:40000]}\n\nQuestion: {prompt}"
    
    st.toast("🧠 Routing to GEMINI (Precision Mode)", icon="🟣")
    
    # List of models to try in order (if one fails, try the next)
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            continue # Try next model quietly
            
    return "❌ GEMINI ERROR: Could not connect to any Gemini model. Check API Key or Region."

# --- 4. THE AGENTIC ROUTER (IMPROVED) ---
def agentic_router(user_input, context_text=""):
    """Decides which model to use based on complexity and length."""
    if not genai:
        return call_llm_groq(user_input, context_text), "GROQ"

    # --- LOGIC 1: LENGTH TRAP ---
    if len(user_input) > 200:
        return call_llm_gemini(user_input, context_text), "GEMINI"

    # --- LOGIC 2: KEYWORD TRAP ---
    deep_keywords = ["summarize", "explain", "analyze", "detailed", "study plan", "math", "code"]
    if any(keyword in user_input.lower() for keyword in deep_keywords):
        return call_llm_gemini(user_input, context_text), "GEMINI"

    # --- LOGIC 3: LLM DECISION ---
    router_prompt = f"""
    Analyze request: "{user_input}"
    Output ONLY: 'GROQ' (simple/fast) or 'GEMINI' (complex/reasoning).
    """
    try:
        # Use a safe fallback model for the router too
        router_model = genai.GenerativeModel("gemini-pro") 
        decision = router_model.generate_content(router_prompt).text.strip().upper()
    except:
        decision = "GROQ"

    if "GEMINI" in decision:
        return call_llm_gemini(user_input, context_text), "GEMINI"
    else:
        return call_llm_groq(user_input, context_text), "GROQ"

# --- 5. ADAPTIVE THEME STYLING ---
st.markdown("""
<style>
    .stButton button {
        background-color: #00FFA3 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    .badge-groq {
        background: linear-gradient(135deg, #00FFA3 0%, #00CC82 100%);
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .badge-gemini {
        background: linear-gradient(135deg, #9D4EDD 0%, #7B2CBF 100%);
        color: #FFFFFF;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .response-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 6. MAIN UI ---
st.title("👋 Welcome to TaleemAI")
st.markdown("### I am your Agentic Multi-Modal Tutor")
st.caption("🎯 Architecture: Self-Healing Router | In-Memory RAG | Offline TTS")

# Sidebar
with st.sidebar:
    st.header("📁 Study Materials")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    pdf_context = ""
    if uploaded_file:
        with st.spinner("📚 Ingesting PDF..."):
            pdf_context = extract_text_from_pdf(uploaded_file)
        st.success(f"✅ {uploaded_file.name} Loaded")
        st.info(f"📄 {len(pdf_context)} chars extracted")

# Main content
user_query = st.text_area("Ask your AI Tutor:", height=100, placeholder="e.g. Summarize this PDF...")

if st.button("🚀 Run Agent"):
    if not user_query:
        st.warning("⚠️ Please type a question.")
    elif not GROQ_API_KEY or not GOOGLE_API_KEY:
        st.error("🚨 API Keys Missing!")
    else:
        with st.spinner("🤖 Orchestrating Agents..."):
            # Route & Execute
            response_text, route_used = agentic_router(user_query, pdf_context)
            
            # Show Badge
            if route_used == "GROQ":
                st.markdown('<div class="badge-groq">⚡ Routed to Groq</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-gemini">🧠 Routed to Gemini</div>', unsafe_allow_html=True)
            
            # Display Text
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown(response_text)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Audio
            st.markdown("---")
            with st.spinner("🎤 Generating audio..."):
                # Using Offline TTS now
                audio_file = text_to_speech(response_text[:500])
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')