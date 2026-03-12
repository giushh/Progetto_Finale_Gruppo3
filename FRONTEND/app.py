import os
import keras
from keras import layers
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.express as px
from streamlit.components.v1 import html
from pathlib import Path


@keras.saving.register_keras_serializable(package="Custom")
class ColorJitter(layers.Layer):
    def __init__(self, brightness=0.08, contrast=0.12, saturation=0.10, **kwargs):
        super().__init__(**kwargs)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def call(self, x, training=None):
        if training is False or training is None:
            return x
        x = tf.image.random_brightness(x, max_delta=self.brightness)
        x = tf.image.random_contrast(x, lower=1.0 - self.contrast, upper=1.0 + self.contrast)
        x = tf.image.random_saturation(x, lower=1.0 - self.saturation, upper=1.0 + self.saturation)
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
        })
        return config
    
st.set_page_config(
    page_title="Image Recognizer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(PROJECT_ROOT, "RESULT/cifar10_improved_model_V2.keras")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(167, 243, 208, 0.35), transparent 26%),
            radial-gradient(circle at 85% 10%, rgba(187, 247, 208, 0.22), transparent 22%),
            radial-gradient(circle at 50% 85%, rgba(110, 231, 183, 0.12), transparent 30%),
            linear-gradient(135deg, #eef7f1 0%, #e5f3ea 45%, #dcefe5 100%);
        color: #183229;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        text-align: center;
        margin-bottom: 1.8rem;
    }

    .title {
        font-size: 4.2rem;
        line-height: 1;
        font-weight: 800;
        margin-bottom: 0.9rem;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #1f4f3d 0%, #2f6b53 45%, #4e9b79 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        max-width: 980px;
        margin: 0 auto;
        font-size: 1.08rem;
        line-height: 1.7;
        color: #47685a;
    }

    .hero-classes {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.6rem;
    }

    .hero-chip {
        background: rgba(255, 255, 255, 0.58);
        border: 1px solid rgba(76, 114, 92, 0.12);
        box-shadow: 0 8px 24px rgba(63, 102, 78, 0.08);
        color: #2f5142;
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        font-size: 0.9rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }

    .upload-shell {
        max-width: 900px;
        margin: 0 auto 1.3rem auto;
        padding: 1.3rem 1.4rem 1.15rem 1.4rem;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid rgba(77, 116, 92, 0.14);
        box-shadow:
            0 20px 50px rgba(55, 89, 68, 0.10),
            inset 0 1px 0 rgba(255,255,255,0.7);
        backdrop-filter: blur(14px);
    }

    .upload-title {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 800;
        color: #244535;
        margin-bottom: 0.45rem;
    }

    .upload-subtitle {
        text-align: center;
        color: #5a7a6b;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 0.2rem;
    }

    .glass-panel {
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid rgba(77, 116, 92, 0.14);
        border-radius: 26px;
        padding: 1.2rem 1.2rem 1.25rem 1.2rem;
        box-shadow:
            0 18px 45px rgba(55, 89, 68, 0.08),
            inset 0 1px 0 rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #234736;
        margin-bottom: 0.2rem;
    }

    .section-caption {
        font-size: 0.96rem;
        color: #5d7a6c;
        margin-bottom: 1rem;
    }

    .image-card {
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(77, 116, 92, 0.12);
        box-shadow: 0 14px 36px rgba(55, 89, 68, 0.10);
    }

    .mini-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }

    .mini-pill {
        padding: 0.58rem 0.95rem;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(219, 241, 228, 0.95), rgba(208, 235, 218, 0.92));
        border: 1px solid rgba(88, 132, 106, 0.14);
        color: #2d4e3f;
        font-size: 0.92rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(55, 89, 68, 0.06);
    }

    .pred-item {
        background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(245,252,247,0.66));
        border: 1px solid rgba(82, 125, 99, 0.12);
        box-shadow: 0 10px 24px rgba(55, 89, 68, 0.06);
        border-radius: 20px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }

    .pred-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .pred-left {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        min-width: 0;
    }

    .rank-dot {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #8fd3b0 0%, #5eaa82 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        flex-shrink: 0;
        box-shadow: 0 8px 18px rgba(71, 130, 98, 0.20);
    }

    .pred-label {
        font-size: 1rem;
        font-weight: 800;
        color: #224434;
        text-transform: capitalize;
    }

    .pred-sub {
        font-size: 0.85rem;
        color: #678273;
        margin-top: 0.12rem;
    }

    .pred-value {
        text-align: right;
        flex-shrink: 0;
        font-size: 1.15rem;
        font-weight: 800;
        color: #1f4031;
    }

    .summary-box {
        margin-top: 1rem;
        padding: 1rem 1.05rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(226, 244, 234, 0.95), rgba(215, 237, 223, 0.92));
        border: 1px solid rgba(83, 126, 100, 0.12);
        color: #29473a;
        line-height: 1.65;
        font-size: 0.98rem;
        box-shadow: 0 8px 22px rgba(55, 89, 68, 0.05);
    }

    .summary-box strong {
        color: #1f4031;
    }

    .empty-box {
        max-width: 920px;
        margin: 1.25rem auto 0 auto;
        padding: 2.2rem 1.5rem;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid rgba(77, 116, 92, 0.14);
        box-shadow:
            0 18px 45px rgba(55, 89, 68, 0.08),
            inset 0 1px 0 rgba(255,255,255,0.65);
        text-align: center;
        backdrop-filter: blur(14px);
    }

    .empty-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #244535;
        margin-bottom: 0.6rem;
    }

    .empty-text {
        max-width: 760px;
        margin: 0 auto;
        font-size: 1rem;
        line-height: 1.7;
        color: #5a7a6b;
    }

    .stFileUploader > label {
        display: none;
    }

    .stFileUploader section {
        background: linear-gradient(180deg, rgba(251,255,252,0.98), rgba(240,249,243,0.95)) !important;
        border: 2px dashed rgba(97, 149, 118, 0.38) !important;
        border-radius: 24px !important;
        padding: 0.9rem 1rem !important;
        min-height: 120px !important;
        transition: all 0.2s ease !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.75) !important;
    }

    .stFileUploader section:hover {
        border-color: rgba(70, 125, 93, 0.60) !important;
        background: linear-gradient(180deg, rgba(250,255,251,1), rgba(235,247,239,0.96)) !important;
    }
            
    .stFileUploader section.drag-active,
        [data-testid="stFileUploaderDropzone"].drag-active,
        [data-testid="stFileUploaderDropzone"] section.drag-active {
            border-color: #3f8f68 !important;
            background: linear-gradient(180deg, rgba(235, 252, 242, 1), rgba(214, 242, 224, 0.98)) !important;
            box-shadow:
                0 0 0 4px rgba(100, 181, 130, 0.18),
                inset 0 1px 0 rgba(255,255,255,0.85) !important;
            transform: scale(1.01);
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 0.4rem 0.2rem !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.35rem !important;
        text-align: center !important;
        width: 100% !important;
    }
            
    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 0 auto !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        display: block !important;
        margin: 0 auto !important;
        transform: translateX(0) translateY(32px) !important;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    [data-testid="stFileUploaderFileName"] {
        color: #1f4031 !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] div {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #264736 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 0.93rem !important;
        color: #5c7a6b !important;
    }

    .stFileUploader button {
        border-radius: 14px !important;
        border: 1px solid rgba(85, 126, 101, 0.18) !important;
        background: #e8f5ec !important;
        color: #234736 !important;
        font-weight: 700 !important;
        margin-left: -14px !important;
    }

    .stFileUploader button:hover {
        background: #dcefe3 !important;
        color: #1d3e2f !important;
        border-color: rgba(85, 126, 101, 0.28) !important;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.46);
        border: 1px solid rgba(77, 116, 92, 0.12);
        border-radius: 18px;
        overflow: hidden;
    }

    [data-testid="stExpander"] details summary p {
        color: #2e5142 !important;
        font-weight: 700 !important;
    }

    .stAlert {
        border-radius: 16px !important;
    }

    div[data-testid="stProgressBar"] > div {
        background-color: rgba(125, 160, 140, 0.16) !important;
    }

    div[data-testid="stProgressBar"] div div {
        background: linear-gradient(90deg, #9ad7b7 0%, #5ca67e 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model not found at: {MODEL_PATH}")
        return None
    try:
        # --- MODIFICA QUESTA RIGA ---
        return keras.models.load_model(
            MODEL_PATH, 
            custom_objects={"ColorJitter": ColorJitter}
        )
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def process_and_predict(image, model):
    img = image.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    top_3_indices = np.argsort(preds)[-3:][::-1]
    top_results = [(CLASS_NAMES[i], float(preds[i])) for i in top_3_indices]
    all_results = [(CLASS_NAMES[i], float(preds[i])) for i in range(len(CLASS_NAMES))]

    return top_results, all_results, img_array[0]


def confidence_label(prob):
    if prob >= 0.80:
        return "Very high confidence"
    if prob >= 0.50:
        return "High confidence"
    if prob >= 0.20:
        return "Moderate confidence"
    return "Low confidence"


def make_donut_chart(all_predictions):
    df = pd.DataFrame(all_predictions, columns=["Class", "Probability"])
    df = df.sort_values("Probability", ascending=False)

    color_map = {
        "airplane": "#d8f0df",
        "automobile": "#cae8d2",
        "bird": "#bfe3c8",
        "cat": "#b4ddbf",
        "deer": "#a8d7b6",
        "dog": "#9cd0ad",
        "frog": "#67b587",
        "horse": "#8dc6a1",
        "ship": "#cfeede",
        "truck": "#7dbd93",
    }

    fig = px.pie(
        df,
        names="Class",
        values="Probability",
        hole=0.52,
        color="Class",
        color_discrete_map=color_map
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#eef7f1", width=2))
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#355646")
        ),
        font=dict(family="Inter, sans-serif", color="#29473a")
    )

    return fig


def render_top3_streamlit(top_predictions):
    st.subheader("Top predictions")

    for idx, (label, prob) in enumerate(top_predictions, start=1):
        st.markdown('<div class="pred-item">', unsafe_allow_html=True)

        left, right = st.columns([0.72, 0.28], gap="small")

        with left:
            st.markdown(
                f"""
                <div class="pred-head">
                    <div class="pred-left">
                        <div class="rank-dot">{idx}</div>
                        <div>
                            <div class="pred-label">{label}</div>
                            <div class="pred-sub">{confidence_label(prob)}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                f"""
                <div class="pred-value">{prob:.1%}</div>
                """,
                unsafe_allow_html=True
            )

        st.progress(float(prob))
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    chips_html = "".join([f'<div class="hero-chip">{name}</div>' for name in CLASS_NAMES])

    st.markdown(
        f"""
        <div class="hero">
            <div class="title">Image Recognizer</div>
            <div class="subtitle">
                Upload an image and let the model determine which category it belongs to.<br>
                Supported classes:</strong>
            </div>
            <div class="hero-classes">
                {chips_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model = get_model()
    if not model:
        st.stop()

    st.markdown(
        """
        <div class="upload-shell">
            <div class="upload-title">Upload an image for classification</div>
            <div class="upload-subtitle">
                Drag and drop a file or browse your device to analyze the image.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 2.4, 1])
    with c2:
        html(
            """
            <script>
            const attachDragEffect = () => {
                const dropzone =
                    window.parent.document.querySelector('[data-testid="stFileUploaderDropzone"]') ||
                    window.parent.document.querySelector('.stFileUploader section');

                if (!dropzone || dropzone.dataset.dragHooked === "true") return;

                dropzone.dataset.dragHooked = "true";

                let dragCounter = 0;

                const activate = () => {
                    dropzone.classList.add("drag-active");
                    const section = dropzone.closest("section") || dropzone.querySelector("section");
                    if (section) section.classList.add("drag-active");
                };

                const deactivate = () => {
                    dropzone.classList.remove("drag-active");
                    const section = dropzone.closest("section") || dropzone.querySelector("section");
                    if (section) section.classList.remove("drag-active");
                };

                ["dragenter", "dragover"].forEach(evt => {
                    dropzone.addEventListener(evt, (e) => {
                        e.preventDefault();
                        dragCounter++;
                        activate();
                    });
                });

                ["dragleave"].forEach(evt => {
                    dropzone.addEventListener(evt, (e) => {
                        e.preventDefault();
                        dragCounter = Math.max(0, dragCounter - 1);
                        if (dragCounter === 0) deactivate();
                    });
                });

                ["drop"].forEach(evt => {
                    dropzone.addEventListener(evt, (e) => {
                        e.preventDefault();
                        dragCounter = 0;
                        deactivate();
                    });
                });
            };

            const observer = new MutationObserver(() => attachDragEffect());
            observer.observe(window.parent.document.body, { childList: true, subtree: true });

            attachDragEffect();
            </script>
            """,
            height=0,
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        top_predictions, all_predictions, tech_img = process_and_predict(image, model)
        best_label, best_prob = top_predictions[0]

        left_col, right_col = st.columns([1, 1.12], gap="large")

        with left_col:
            st.markdown(
                """
                <div class="glass-panel">
                    <div class="section-title">Input Image</div>
                    <div class="section-caption">Uploaded sample used for inference</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="mini-stats">
                    <div class="mini-pill">Top class: {best_label}</div>
                    <div class="mini-pill">Confidence: {best_prob:.1%}</div>
                    <div class="mini-pill">CNN input: 32×32</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Technical preview"):
                st.image(tech_img, caption="Normalized model input", width=220)
                st.info(
                    f"Statistics: Min={tech_img.min():.2f}, Max={tech_img.max():.2f}, Mean={tech_img.mean():.2f}"
                )

        with right_col:
            st.markdown(
                """
                <div class="glass-panel">
                    <div class="section-title">Analysis Results</div>
                    <div class="section-caption">Top predictions and probability distribution</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            render_top3_streamlit(top_predictions)

            fig = make_donut_chart(all_predictions)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""
                <div class="summary-box">
                    <strong>Predicted class:</strong> {best_label}<br>
                    Based on the visual features extracted from the image, the model considers
                    this sample most similar to the <strong>{best_label}</strong> category,
                    with a confidence score of <strong>{best_prob:.1%}</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )

            

    else:
        st.markdown(
            """
            <div class="empty-box">
                <div class="empty-title">Ready to analyze an image</div>
                <div class="empty-text">
                    Upload a file using the area above and the model will estimate which supported category
                    best matches the image content.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()