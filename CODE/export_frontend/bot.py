import discord
from discord.ext import commands
import tensorflow as tf
import numpy as np
import asyncio
from PIL import Image
import io

# Configurazione Bot
TOKEN = 'MTQ4MDU1NzkxMjc5NDMzNzMzMA.Gj4_x6.4UURZ4sS9HQPFgUOZOnCYgw9ZifNDYXKUrZCdU'
MODEL_PATH = 'cifar10_improved_model.keras'

# Caricamento del modello
model = tf.keras.models.load_model(MODEL_PATH)

# Classi CIFAR-10
class_names = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 
               'cane', 'rana', 'cavallo', 'nave', 'camion']

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def prepare_image(image_bytes):
    # Apre l'immagine, la ridimensiona a 32x32 (formato CIFAR-10) e la normalizza
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@bot.event
async def on_ready():
    print(f'Bot collegato come {bot.user.name}')

@bot.command()
async def classifica(ctx):
    # Controlla se c'è un allegato
    if not ctx.message.attachments:
        await ctx.send("Per favore, allega un'immagine al messaggio con il comando !classifica.")
        return

    attachment = ctx.message.attachments[0]
    if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
        await ctx.send("Formato file non supportato. Invia un'immagine (jpg/png).")
        return

    # Messaggio iniziale con barra di caricamento simulata
    msg = await ctx.send("Analisi in corso... \n[░░░░░░░░░░] 0%")
    
    try:
        # Download immagine
        image_bytes = await attachment.read()
        
        # Simulazione avanzamento barra
        await asyncio.sleep(0.5)
        await msg.edit(content="Analisi in corso... \n[████░░░░░░] 40%")
        
        # Preprocessing e Predizione
        img_array = prepare_image(image_bytes)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        await asyncio.sleep(0.5)
        await msg.edit(content="Analisi in corso... \n[████████░░] 80%")
        
        label = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        await asyncio.sleep(0.5)
        await msg.edit(content=f"✅ **Analisi Completata!**\n[██████████] 100%\n\n**Risultato:** L'immagine sembra contenere un **{label}** (confidenza: {confidence:.2f}%)")

    except Exception as e:
        await msg.edit(content=f"Si è verificato un errore durante l'analisi: {e}")

@bot.command()
async def info(ctx):
    embed = discord.Embed(
        title="Informazioni Bot Visione Artificiale",
        description="Questo bot utilizza un modello di Deep Learning (CNN) per riconoscere oggetti nelle immagini.",
        color=discord.Color.blue()
    )
    embed.add_field(name="!classifica", value="Invia un'immagine con questo comando per farla analizzare dal modello.", inline=False)
    embed.add_field(name="Capacità", value="Il modello riconosce: aerei, auto, uccelli, gatti, cervi, cani, rane, cavalli, navi e camion.", inline=False)
    embed.add_field(name="Nota", value="Le immagini vengono ridimensionate a 32x32 pixel per l'elaborazione.", inline=False)
    await ctx.send(embed=embed)

@bot.event
async def on_message(message):
    print(f"Messaggio ricevuto: {message.content}") # Vedrai nel terminale cosa legge il bot
    await bot.process_commands(message) # Senza questo i comandi ! non funzionano

bot.run(TOKEN)