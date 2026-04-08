# ## Implementazione di un Bot Discord per la Classificazione CIFAR-10
# In questa sezione integriamo il modello addestrato con un Bot Discord. 
# Il bot riceverà un'immagine, la processerà e restituirà la classe predetta.


# Installazione delle librerie necessarie
#!pip install discord.py tensorflow pillow nest-asyncio

import nest_asyncio
import io
import numpy as np
from PIL import Image
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv 
from pathlib import Path
import sys

# Aggiungi la root del progetto al sys.path PRIMA di importare moduli personalizzati
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import keras
from CODE.utils import ColorJitter


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
MODEL_PATH = os.path.join(PROJECT_ROOT, "RESULT", "cifar10_improved_model_V2.keras")


try:
    # Aggiungiamo compile=False per ignorare l'ottimizzatore PyTorch incompatibile
    model = keras.models.load_model(MODEL_PATH, compile=False,custom_objects={"ColorJitter": ColorJitter})
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

MAX_IMMAGINI = 5
FORMATI_CONSENTITI = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

@bot.command()
async def classifica(ctx):
    # Controllo 1: almeno un allegato
    if not ctx.message.attachments:
        await ctx.send("⚠️ Allega almeno una foto!")
        return

    # Controllo 2: limite massimo immagini
    if len(ctx.message.attachments) > MAX_IMMAGINI:
        await ctx.send(
            f"⚠️ Puoi inviare al massimo **{MAX_IMMAGINI} immagini** alla volta. "
            f"Hai allegato {len(ctx.message.attachments)}."
        )
        return

    # Iterazione su tutti gli allegati
    for i, attachment in enumerate(ctx.message.attachments, start=1):

        # Controllo 3: formato file valido
        estensione = Path(attachment.filename).suffix.lower()
        if estensione not in FORMATI_CONSENTITI:
            await ctx.send(
                f"⚠️ L'immagine **{attachment.filename}** non è supportata.\n"
                f"Formati accettati: `{', '.join(FORMATI_CONSENTITI)}`"
            )
            continue  # Salta e passa alla prossima

        loading_msg = await ctx.send(f"⌛ [{i}/{len(ctx.message.attachments)}] Inizializzazione scansione...")

        try:
            # Step 1 — Acquisizione e pre-processing
            await loading_msg.edit(content=f"📡 [{i}] `███░░░░░░░` Acquisizione e ricampionamento...")
            image_bytes = await attachment.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            small_img = pil_img.resize((32, 32), Image.Resampling.LANCZOS)
            display_img = small_img.resize((400, 400), Image.Resampling.NEAREST)

            image_binary = io.BytesIO()
            display_img.save(image_binary, 'PNG')
            image_binary.seek(0)
            discord_file = discord.File(fp=image_binary, filename=f"pixel_analysis_{i}.png")

            processed_img = prepare_image(image_bytes)

            # Step 2 — Inferenza
            await loading_msg.edit(content=f"🧠 [{i}] `███████░░░` Analisi neurale in corso...")
            preds = model(processed_img, training=False)
            probs = preds[0].numpy()

            top3_indices = np.argsort(probs)[-3:][::-1]
            index_primario = top3_indices[0]
            confidenza_primaria = probs[index_primario] * 100

            await loading_msg.edit(content=f"✅ [{i}] `██████████` Scansione completata!")

            # Step 3 — Costruzione embed
            embed = discord.Embed(
                title=f"🔍 Analisi CIFAR-10 — Immagine {i}/{len(ctx.message.attachments)}",
                color=0x3498db
            )
            embed.set_image(url=f"attachment://pixel_analysis_{i}.png")
            embed.add_field(
                name="Oggetto Identificato",
                value=f"✨ **{class_names[index_primario].upper()}**",
                inline=False
            )
            embed.add_field(
                name="Grado di Sicurezza",
                value=f"📈 {confidenza_primaria:.2f}%",
                inline=True
            )

            testo_top3 = ""
            for j, idx in enumerate(top3_indices):
                nome = class_names[idx].capitalize()
                p = probs[idx] * 100
                testo_top3 += f"**{j+1}.** {nome}: `{p:.2f}%`\n"
            embed.add_field(name="📊 Top 3 Probabilità", value=testo_top3, inline=False)
            embed.set_footer(text="Vista AI: 32x32 (Upscaled) | MaGMI Image Recognizer")

            # Step 4 — Invio risultato
            await loading_msg.delete()
            await ctx.send(file=discord_file, embed=embed)

        except Exception as e:
            if 'loading_msg' in locals():
                await loading_msg.edit(content=f"❌ Errore durante la scansione dell'immagine {i}: {e}")
            else:
                await ctx.send(f"❌ Errore critico sull'immagine {i}: {e}")

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


""" if __name__ == "__main__":
    main()
 """