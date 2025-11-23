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

# --- 1. SECRETS SETUP (Fixed for Hugging Face) ---
# Try to get keys from Environment (Hugging Face Secrets) or Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

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

def text_to_speech(text):
    """Converts text to audio using gTTS (With Crash Guard)."""
    try:
        # Create audio in memory
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp
    except Exception as e:
        # If audio fails (Rate Limit), we return None so the App doesn't crash
        print(f"Audio Error: {e}")
        return None

# --- 3. LLM FUNCTIONS (Improved with Better Error Handling) ---
def call_llm_groq(prompt, context=""):
    """Calls Groq (Llama 3.1) for fast generation."""
    if not groq_client:
        return "⚠️ GROQ NOT READY: Check API Key.", False
    
    full_prompt = prompt
    if context:
        full_prompt = f"Context: {context[:6000]}\n\nTask: {prompt}"
    
    st.toast("⚡ Routing to GROQ (Speed Mode)", icon="🟢")
    
    try:
        out = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=2048,
            temperature=0.7
        )
        return out.choices[0].message.content, True
    except Exception as e:
        return f"❌ GROQ ERROR: {str(e)}", False

def call_llm_gemini(prompt, context=""):
    """Calls Gemini with Improved Fallback Logic and Error Handling."""
    if not genai:
        return "⚠️ GEMINI NOT READY: Check API Key.", False
    
    full_prompt = prompt
    if context:
        # Increased context limit for deep reading
        full_prompt = f"Context: {context[:40000]}\n\nQuestion: {prompt}"
    
    st.toast("🧠 Routing to GEMINI (Precision Mode)", icon="🟣")
    
    # UPDATED: Better model priority list (free tier models first)
    models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest", 
        "gemini-1.5-flash-8b",
        "gemini-pro",
        "gemini-1.0-pro"
    ]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
            )
            return response.text, True
        except Exception as e:
            error_msg = str(e)
            # Check if it's a quota error
            if "429" in error_msg or "quota" in error_msg.lower():
                st.warning(f"⚠️ {model_name} quota exceeded, trying next model...")
                continue
            elif "404" in error_msg:
                # Model doesn't exist, try next
                continue
            else:
                # Other error, try next model
                st.warning(f"⚠️ {model_name} failed: {error_msg[:100]}")
                continue
    
    # If all Gemini models fail, return error
    return "❌ GEMINI ERROR: All models exhausted or quota exceeded. Please try again later or use a different API key.", False

# --- 4. THE AGENTIC ROUTER (Enhanced with Fallback) ---
def agentic_router(user_input, context_text=""):
    """Decides model based on length and intent with automatic fallback."""
    
    # If only Groq is available, use it
    if not genai or not GOOGLE_API_KEY:
        return call_llm_groq(user_input, context_text)
    
    # If only Gemini is available, use it
    if not groq_client or not GROQ_API_KEY:
        return call_llm_gemini(user_input, context_text)

    # Rule 1: Length Trap (Long queries -> Gemini first, fallback to Groq)
    if len(user_input) > 200: 
        response, success = call_llm_gemini(user_input, context_text)
        if success:
            return response, "GEMINI"
        else:
            st.info("Falling back to Groq...")
            response, _ = call_llm_groq(user_input, context_text)
            return response, "GROQ (Fallback)"

    # Rule 2: Keyword Trap (Deep words -> Gemini first, fallback to Groq)
    keywords = ["summarize", "explain", "study", "math", "code", "detailed", "analyze"]
    if any(k in user_input.lower() for k in keywords):
        response, success = call_llm_gemini(user_input, context_text)
        if success:
            return response, "GEMINI"
        else:
            st.info("Falling back to Groq...")
            response, _ = call_llm_groq(user_input, context_text)
            return response, "GROQ (Fallback)"

    # Rule 3: AI Decision (with error handling)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(
            f"Classify this query as needing either 'GROQ' (fast, simple) or 'GEMINI' (detailed, complex): '{user_input}'. Reply with only one word: GROQ or GEMINI"
        )
        if "GEMINI" in res.text.upper():
            response, success = call_llm_gemini(user_input, context_text)
            if success:
                return response, "GEMINI"
            else:
                st.info("Falling back to Groq...")
                response, _ = call_llm_groq(user_input, context_text)
                return response, "GROQ (Fallback)"
    except:
        pass
    
    # Default: Use Groq for speed
    response, success = call_llm_groq(user_input, context_text)
    return response, "GROQ"

# --- 5. RESTORED CSS & THEME ---
st.markdown("""
<style>
    /* RESTORED: Button styling */
    .stButton button {
        background-color: #00FFA3 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #00CC82 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 255, 163, 0.4) !important;
    }
    
    /* RESTORED: Badges */
    .badge-groq {
        background: linear-gradient(135deg, #00FFA3 0%, #00CC82 100%);
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .badge-gemini {
        background: linear-gradient(135deg, #9D4EDD 0%, #7B2CBF 100%);
        color: #FFFFFF;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .badge-fallback {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: #FFFFFF;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* RESTORED: Response Box */
    .response-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 6. MAIN UI (Restored) ---
st.title("Welcome to TaleemAI")
st.markdown("### I am your Agentic Multi-Modal Tutor")
st.caption("Architecture: Self-Healing Router | In-Memory RAG | Audio TTS")

# Show API Status
with st.sidebar:
    st.header("🔌 API Status")
    if GROQ_API_KEY:
        st.success("Groq Connected")
    else:
        st.error("❌ Groq Not Connected")
    
    if GOOGLE_API_KEY:
        st.success("Gemini Connected")
    else:
        st.error("❌ Gemini Not Connected")
    
    st.markdown("---")
    st.header("📁 Study Materials")
    st.markdown("##### Upload PDF")
    uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
    
    pdf_context = ""
    if uploaded_file:
        with st.spinner("📚 Ingesting & Indexing PDF..."):
            pdf_context = extract_text_from_pdf(uploaded_file)
        st.success(f"✅ {uploaded_file.name}")
        st.info(f"📄 PDF Ingested ({len(pdf_context)} chars)")
        st.markdown("---")
        if st.button("Clear Conversation"):
            st.rerun()

# Main Chat Area
st.markdown("### 👉 Upload a PDF to start RAG mode or just ask a question below.")

user_query = st.text_area(
    "Ask your AI Tutor:", 
    height=120, 
    placeholder="e.g. Summarize the physics formulas in this PDF...",
    key="query_input"
)

if st.button("🚀 Run Agent"):
    if not user_query:
        st.warning("⚠️ Please type a question first.")
    elif not GROQ_API_KEY and not GOOGLE_API_KEY:
        st.error("🚨 API Keys Missing! Please add at least one API key (GROQ_API_KEY or GOOGLE_API_KEY) in Settings > Secrets.")
    else:
        with st.spinner("🤖 Orchestrating Agents..."):
            # Route & Execute
            response_text, route_used = agentic_router(user_query, pdf_context)
            
            # Show routing badge
            if "GROQ" in route_used:
                if "Fallback" in route_used:
                    st.markdown('<div class="badge-fallback">⚡ Routed to Groq (Fallback Mode)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-groq">⚡ Routed to Groq (Speed)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-gemini">🧠 Routed to Gemini (Precision)</div>', unsafe_allow_html=True)
            
            # Display response
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown(response_text)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Audio generation (With Crash Guard)
            st.markdown("---")
            st.markdown("### 🎧 Audio Ready")
            
            with st.spinner("🎤 Generating audio lecture..."):
                audio_file = text_to_speech(response_text[:500])
                if audio_file:
                    st.markdown("##### 🔊 Listen to Audio Lecture")
                    st.audio(audio_file, format='audio/mp3')
                else:
                    st.warning("Audio generation skipped (Rate Limit or Error). Text response is preserved.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by TEAM-TM-2300")