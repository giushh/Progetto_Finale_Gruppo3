# ## Implementazione di un Bot Discord per la Classificazione CIFAR-10
# In questa sezione integriamo il modello addestrato con un Bot Discord. 
# Il bot riceverà un'immagine, la processerà e restituirà la classe predetta.


# Installazione delle librerie necessarie
#!pip install discord.py tensorflow pillow nest-asyncio

import nest_asyncio
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import keras
from PIL import Image

# Patch necessaria per far girare il loop di Discord dentro Jupyter
nest_asyncio.apply()  #Ho incluso nest_asyncio perché i file Jupyter hanno spesso conflitti con i loop di Discord.

load_dotenv()
TOKEN = os.getenv('TOKEN')

if not TOKEN:
    print("❌ ERRORE CRITICO: Il Token di Discord non è stato trovato!")
    print("Assicurati di avere un file chiamato '.env' (senza nome prima del punto) nella stessa cartella, contenente: TOKEN=il_tuo_token_qui")
    exit() # Ferma il programma per evitare l'errore di Discord


# ##### 1. Caricamento del Modello e Definizione Classi
# Carichiamo il file `keras` generato durante la fase di training e definiamo le 10 etichette del dataset CIFAR-10.

# Percorso relativo dalla cartella bot/ a RESULT/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = os.path.join(PROJECT_ROOT, "RESULT", "cifar10_improved_model.keras")


try:
    # Aggiungiamo compile=False per ignorare l'ottimizzatore PyTorch incompatibile
    model = keras.models.load_model(MODEL_PATH, compile=False)
    if model:
        print("✅ Modello caricato correttamente (modalità inferenza)!")

    # Visualizziamo la struttura per essere sicuri che sia integro
        model.summary()

except Exception as e:
    print(f"❌ Errore nel caricamento: {e}")

class_names = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 
               'cane', 'rana', 'cavallo', 'nave', 'camion']


# # Preprocessing

# ## 2. Funzione di Pre-processing
# Le immagini inviate su Discord possono avere qualsiasi dimensione. La funzione `prepare_image` si occupa di:
# 1. Convertire l'immagine in RGB.
# 2. Ridimensionarla a 32x32 pixel.
# 3. Normalizzare i valori dei pixel tra 0 e 1.


def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)


# # Configurazione e Comandi Bot

# ### 3. Configurazione del Bot Discord
# Definiamo il prefisso del comando (`!`) e la logica per gestire l'allegato. 
# Il bot scaricherà l'immagine, userà il modello per la predizione e risponderà all'utente.


# Configurazione permessi (Intents)
intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.command()
async def classifica(ctx):
    # Controllo iniziale: esiste un allegato?
    if not ctx.message.attachments:
        await ctx.send("⚠️ Allega una foto!")
        return
    attachment = ctx.message.attachments[0]
    # 1. Messaggio di stato iniziale
    loading_msg = await ctx.send("⌛ Inizializzazione scansione...")

    try:
    # 2. Acquisizione e Pre-processing fisico
        await loading_msg.edit(content="📡 `███░░░░░░░` Acquisizione e ricampionamento...")
        # Lettura immagine originale
        image_bytes = await attachment.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # --- LOGICA DI RIDIMENSIONAMENTO ---
        #Creiamo la versione 32x32 per il modello (Uso Lanczos per la qualità del segnale)
        small_img = pil_img.resize((32, 32), Image.Resampling.LANCZOS)
        # Ingrandiamo la versione 32x32 a 400x400 per la visualizzazione Discord
        # Usiamo NEAREST per mantenere i pixel nitidi (effetto "pixel art")
        display_img = small_img.resize((400, 400), Image.Resampling.NEAREST)
        # Salvataggio in un buffer di memoria per l'invio
        image_binary = io.BytesIO()
        display_img.save(image_binary, 'PNG')
        image_binary.seek(0)
        discord_file = discord.File(fp=image_binary, filename="pixel_analysis.png")

        # Prepariamo l'immagine per l'inferenza (passiamo i bytes al modello)
        processed_img = prepare_image(image_bytes)
        # 3. Fase di Inferenza (Analisi Neurale)
        await loading_msg.edit(content="🧠 `███████░░░` Analisi neurale in corso...")
        preds = model(processed_img, training=False)
        probs = preds[0].numpy()

        # Calcolo Top 3 Probabilità

        top3_indices = np.argsort(probs)[-3:][::-1]
        index_primario = top3_indices[0]
        confidenza_primaria = probs[index_primario] * 100

       # 4. Completamento e Invio Risultati
        await loading_msg.edit(content="✅ `██████████` Scansione completata!")

        # Costruzione dell'Embed
        embed = discord.Embed(title="🔍 Analisi CIFAR-10", color=0x3498db)

       # Impostiamo l'immagine ingrandita come immagine principale
        embed.set_image(url="attachment://pixel_analysis.png")

       # Risultato principale (Top 1)
        embed.add_field(name="Oggetto Identificato", value=f"✨ **{class_names[index_primario].upper()}**", inline=False)
        embed.add_field(name="Grado di Sicurezza", value=f"📈 {confidenza_primaria:.2f}%", inline=True)

       # Costruzione testo per Top 3

        testo_top3 = ""
        for i, idx in enumerate(top3_indices):

            nome = class_names[idx].capitalize()
            p = probs[idx] * 100
            testo_top3 += f"**{i+1}.** {nome}: `{p:.2f}%`\n"
        embed.add_field(name="📊 Top 3 Probabilità", value=testo_top3, inline=False)
        embed.set_footer(text="Vista AI: 32x32 (Upscaled) | MaGMI Image Recognizer")

        # Rimuoviamo il caricamento e inviamo file + embed

        await loading_msg.delete()
        await ctx.send(file=discord_file, embed=embed)

    except Exception as e:

        # Gestione errori robusta
        if 'loading_msg' in locals():
            await loading_msg.edit(content=f"❌ Errore durante la scansione: {e}")
        else:
            await ctx.send(f"❌ Errore critico: {e}")


# ### Gestione Errori

# 1. Configurazione Stato Personalizzato
@bot.event
async def on_ready():
    # Imposta lo stato: "In ascolto di !info"
    activity = discord.Activity(type=discord.ActivityType.listening, name="!info")
    await bot.change_presence(status=discord.Status.online, activity=activity)
    print(f'✅ Bot online come: {bot.user}')

# 2. Gestione Errori Globale
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❓ Comando non riconosciuto. Scrivi `!info` per vedere cosa posso fare.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("⚠️ Mancano degli argomenti nel comando.")
    else:
        print(f"Errore non gestito: {error}")


# ### Comando Informativo
# Aggiungiamo un comando per spiegare agli utenti quali sono le 10 categorie che il modello può riconoscere (dataset CIFAR-10) e come utilizzare il bot.


@bot.command()
async def info(ctx):
    # Creiamo un messaggio formattato in modo elegante (Embed)
    descrizione = (
        "Ciao! Sono il Bot del progetto di Classificazione Immagini.\n"
        "Utilizzo una rete neurale (CNN) addestrata sul dataset **CIFAR-10**.\n\n"
        "**Cosa posso fare?**\n"
        "Se mi invii una foto e scrivi `!classifica`, proverò a capire cosa rappresenta.\n\n"
        "**Le mie 10 categorie sono:**\n"
        f"✈️ {class_names[0]}, 🚗 {class_names[1]}, 🐦 {class_names[2]}, 🐱 {class_names[3]}, 🦌 {class_names[4]},\n"
        f"🐶 {class_names[5]}, 🐸 {class_names[6]}, 🐴 {class_names[7]}, 🚢 {class_names[8]}, 🚛 {class_names[9]}.\n\n"
        "**Istruzioni:**\n"
        "Carica una foto e scrivi `!classifica` nel campo del commento!"
    )

    await ctx.send(descrizione)


# # Esecuzione
# Inserisci il tuo Token e avvia la cella. Il bot rimarrà attivo finché la cella è in esecuzione.


print("⌛ Tentativo di connessione a Discord in corso...")
try:
    bot.run(TOKEN)
except Exception as e:
    print(f"❌ Impossibile avviare il bot. Il bot NON è attivo. Dettaglio Errore: {e}")
