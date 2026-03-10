# Installazione delle librerie necessarie
#!pip install discord.py tensorflow pillow nest-asyncio

import nest_asyncio
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import discord
from discord.ext import commands

# Patch necessaria per far girare il loop di Discord dentro Jupyter
nest_asyncio.apply()  #Ho incluso nest_asyncio perché i file Jupyter hanno spesso conflitti con i loop di Discord.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = r'CODE\export_frontend\cifar10_improved_model.keras'


try:
    # Aggiungiamo compile=False per ignorare l'ottimizzatore PyTorch incompatibile
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Modello caricato correttamente (modalità inferenza)!")
    
    # Visualizziamo la struttura per essere sicuri che sia integro
    model.summary()
    
except Exception as e:
    print(f"❌ Errore nel caricamento: {e}")

class_names = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 
               'cane', 'rana', 'cavallo', 'nave', 'camion']

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    # Normalizzazione [0, 1] coerente con la cella 5 di CNN_base
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv # Carica la libreria

# Carica le variabili dal file .env
load_dotenv()

# Configurazione permessi (Intents)
intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix="!", intents=intents)

class_names = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion']

@bot.command()
async def classifica(ctx):
    if not ctx.message.attachments:
        await ctx.send("⚠️ Allega una foto!")
        return

    attachment = ctx.message.attachments[0]
    # Messaggio di caricamento per feedback immediato
    loading_msg = await ctx.send("⌛ Analisi in corso...")
    
    try:
        image_bytes = await attachment.read()
        processed_img = prepare_image(image_bytes)
        
        # INFERENZA: training=False disattiva RandomRotation/Flip del modello
        preds = model(processed_img, training=False)
        
        # Il modello restituisce probabilità (Softmax è già presente)
        probs = preds[0].numpy()
        index = np.argmax(probs)
        confidenza = probs[index] * 100
        
        # Creazione dell'Embed
        embed = discord.Embed(title="🔍 Analisi CIFAR-10", color=0x3498db)
        embed.set_thumbnail(url=attachment.url)
        embed.add_field(name="Oggetto Identificato", value=f"✨ **{class_names[index].upper()}**", inline=False)
        embed.add_field(name="Grado di Sicurezza", value=f"📈 {confidenza:.2f}%", inline=True)
        
        # Diagnostica per il Check-sum dei pixel (deve variare tra immagini diverse)
        p_sum = np.sum(processed_img)
        embed.set_footer(text=f"Check-sum: {p_sum:.2f} | Mode: Inference")

        await loading_msg.delete()
        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"❌ Errore critico: {e}")

@bot.event
async def on_ready():
    # Imposta lo stato: "In ascolto di !classifica"
    activity = discord.Activity(type=discord.ActivityType.listening, name="!classifica")
    await bot.change_presence(status=discord.Status.online, activity=activity)
    print(f'✅ Bot pronto: {bot.user}')

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

from tensorflow.keras.datasets import cifar10

@bot.command()
async def test(ctx):
    try:
        # Carichiamo i dati originali per un test puro
        (_, _), (x_test_raw, y_test_raw) = cifar10.load_data()
        
        # Prendiamo un'immagine a caso (es. la numero 10, che è un aereo)
        sample_idx = 10 
        img_test = x_test_raw[sample_idx]
        true_label = class_names[y_test_raw[sample_idx][0]]
        
        # Preprocessing identico a quello del bot
        img_input = img_test.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # Predizione con training=False
        preds = model(img_input, training=False)
        probs = preds[0].numpy()
        index = np.argmax(probs)
        
        await ctx.send(f"🧪 **Test Interno**:\n"
                       f"Etichetta Reale: `{true_label.upper()}`\n"
                       f"Predizione Modello: `{class_names[index].upper()}`\n"
                       f"Confidenza: `{probs[index]*100:.2f}%`")
        
    except Exception as e:
        await ctx.send(f"❌ Errore nel test: {e}")
