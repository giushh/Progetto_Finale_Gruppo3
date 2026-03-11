import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Gestione percorsi (Risaliamo dalla cartella FRONTEND alla ROOT)
# Se il modello è in ROOT/export_frontend/ e il file è in ROOT/FRONTEND/
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "CODE", "export_frontend", "cifar10_improved_model.keras")

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"Modello non trovato in: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Funzione di predizione
def predict(image):
    # Preprocessing: ridimensiona, normalizza e aggiunge dimensione batch
    img = image.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array[:, :, :, ::-1]

    
    img_to_show = img_array[0]

    # 2. Mostriamo l'anteprima tecnica nel frontend
    with col1: # o dove preferisci nel layout
        st.write("### Input per la CNN (32x32)")
        # Usiamo width='stretch' per vederla grande anche se è solo 32px
        # Passiamo l'array normalizzato: Streamlit riconosce che se i valori sono 0-1 è un'immagine
        st.image(img_to_show, caption="Immagine normalizzata (0.0 - 1.0)", width='stretch')

    # Debug nel terminale: controlla i valori minimi e massimi
    print(f"Min valore: {img_to_show.min()}, Max valore: {img_to_show.max()}")

    # Predizione
    preds = model.predict(img_array)[0]
    print(f"Raw predictions: {preds}") # Guarda i numeri nel terminale

    # Ottieni gli indici delle prime 3 predizioni
    top_3_indices = np.argsort(preds)[-3:][::-1]
    
    # Restituisce una lista di tuple (NomeClasse, Probabilità)
    return [(class_names[i], float(preds[i])) for i in top_3_indices]

# 3. Interfaccia Streamlit
st.set_page_config(page_title="CIFAR-10 Visual Analyzer", page_icon="🖼️")

st.title("CIFAR-10 Visual Analyzer")
st.write("Analisi probabilistica delle classi CIFAR-10 tramite CNN.")

# Widget per caricare l'immagine
uploaded_file = st.file_uploader("Carica immagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Apri e mostra l'immagine caricata
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    
    # Colonne per affiancare immagine e risultati
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Immagine caricata", use_container_width=True)
        
    with col2:
        st.subheader("Top Predictions")
        with st.spinner("Analisi in corso..."):
            top_predictions = predict(image)
            
            # Mostra le predizioni con una barra di progresso
            for class_name, prob in top_predictions:
                st.write(f"**{class_name.capitalize()}**: {prob:.2%}")
                # st.progress accetta float tra 0.0 e 1.0
                st.progress(prob)