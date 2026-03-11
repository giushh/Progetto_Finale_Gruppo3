import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CIFAR-10 Visual Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px border rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #60a5fa;
        transition: transform 0.2s ease;
    }
    
    .prediction-card:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #60a5fa, #a855f7);
    }
    
    /* Customizing file uploader */
    .stFileUploader section {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ASSETS & MODELS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "cifar10_improved_model.keras")
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model not found at: {MODEL_PATH}")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# --- PREDICTION LOGIC ---
def process_and_predict(image, model):
    # Preprocessing
    img = image.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    preds = model.predict(img_array, verbose=0)[0]
    
    # Get top 3
    top_3_indices = np.argsort(preds)[-3:][::-1]
    results = [(CLASS_NAMES[i], float(preds[i])) for i in top_3_indices]
    
    return results, img_array[0]

# --- UI LAYOUT ---
def main():
    st.markdown('<h1 class="main-title">CIFAR-10 Visual Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">State-of-the-art image classification powered by Deep Learning</p>', unsafe_allow_html=True)

    model = get_model()
    if not model:
        st.stop()

    # Upload Section
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        main_col1, main_col2 = st.columns([1, 1], gap="large")
        
        with main_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🖼️ Input Image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with main_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Analysis Results")
            
            with st.spinner("Classifying neural patterns..."):
                top_predictions, tech_img = process_and_predict(image, model)
                
                for label, prob in top_predictions:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #60a5fa;">{label}</span>
                            <span style="font-weight: 800; font-size: 1.1rem;">{prob:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob)
            
            st.divider()
            with st.expander("🛠️ Technical Preview (CNN Input: 32x32)"):
                st.image(tech_img, caption="Normalized Data Stream", width=200)
                st.info(f"Statistics: Min={tech_img.min():.2f}, Max={tech_img.max():.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome State
        st.info("👆 Please upload an image to begin the analysis.")
        
if __name__ == "__main__":
    main()