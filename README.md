# Progetto Finale Gruppo GGICMADFMN Corso PyML
Classificazione Immagini con CNN (CIFAR-10)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Collaboratori: 
- [Gabriele Giuliani](https://github.com/giulianigabbo95/) (Sviluppo Documentazione)
- [Ilaria Cuccaro](https://github.com/giushh/) (Sviluppo Rete Neurale)
- [Marco Aurelio De Felicis](https://github.com/MarcoAurelioDeFelicis) (Sviluppo Front-End Gradio)
- [Mariagrazia Nuzzolese](https://github.com/mariagrazianuzzolese05-svg) (Sviluppo Front-End Discord)

---

## Regole

### Regole di Setup
Per eseguire correttamente il progetto si consiglia di:
- Clonare il repository con git clone https://github.com/giushh/Progetto_Finale_Gruppo3.git o client Desktop con GUI
- Creare un ambiente virtuale (facoltativo):
    - su Linux / macOS con:
        ```bash
            python3 -m venv venv
            source venv/bin/activate
        ```
    - su Windows con
        ```cmd
            python -m venv venv
            venv\Scripts\activate
        ``` 
- Posizionarsi nella cartella mediante GUI o con
    ```Powershell
        cd Progetto_Finale_Gruppo3
    ```
- Utilizzare [Jupyter Notebook](https://jupyter.org/) oppure eseguire il notebook su piattaforme cloud come [Google Colab](https://colab.research.google.com/) caricando su Google Drive i file appena scaricati.
- Assicurarsi che l'ambiente supporti [**Python 3.12**](https://www.python.org/downloads/release/python-3120/).
- Abilitare l'utilizzo della GPU, se disponibile, per ridurre significativamente i tempi di training della rete neurale.
- Installare le librerie richieste indicate nel file `requirements.txt` (se presente) con
    ```python
        pip install -r requirements.txt.
    ```
- Registrarsi a [Kaggle](https://www.kaggle.com/)
- Creare un New API Token
- Scaricare il file `kaggle.json`
- Configurare la API di Kaggle
    - su Linux / macOS con:
    ```bash
        mkdir -p ~/.kaggle
        mv kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
    ```
    - su Windows con:
    ```cmd
        C:\Users\<username>\.kaggle\ 
    ```
- Scaricare il dataset eseguendo
    ``` bash
        kaggle competitions download -c cifar-10 
    ```
    ed estraendo i file nella cartella `/data`
- Verificare che tutto sia configurato correttamente con python train.py

Nota: il progetto è stato sviluppato e testato utilizzando il dataset **CIFAR-10**.

### Regole di Esecuzione (con lettura del codice)
- Eseguire il file frontend
- Caricare un'immagine a piacere
- Attendere l'esito

### Regole di Esecuzione (senza lettura del codice)
- Aggiungere il bot [image recognizer bot](https://discord.com/oauth2/authorize?client_id=1480557912794337330&permissions=8&integration_type=0&scope=bot) al proprio server Discord
- Avviare il bot
- Caricare un'immagine a piacere scrivendo come descrizione il comando:
    ``` Discord
        !classifica
    ```
- Attendere l'esito

---

## Obiettivo 
In questo progetto viene affrontato un problema di classificazione di immagini utilizzando il dataset CIFAR-10.
L'obiettivo è classificare ogni immagine del dataset in una delle 10 classi disponibili.

Le classi includono sia mezzi di trasporto sia animali, quindi il modello deve imparare a riconoscere:
- forme
- colori
- pattern visivi
- caratteristiche distintive degli oggetti

![Immagini di Esempio](https://github.com/giushh/Progetto_Finale_Gruppo3/blob/img_samples.png)

Oltre a costruire una CNN funzionante, il progetto mira anche a ottimizzarne le prestazioni.  
Vengono quindi utilizzate tecniche come:
- normalizzazione dei pixel
- dropout
- batch normalization
- data augmentation
con l'obiettivo di migliorare la capacità di generalizzazione del modello su immagini mai viste prima.

---

## Dataset: CIFAR-10
Il dataset CIFAR-10 è uno dei più utilizzati per iniziare a lavorare con la classificazione di immagini tramite reti neurali.  
Il nome indica:
- CIFAR → nome del dataset
- 10 → numero di classi da riconoscere

### Dimensioni del dataset
Il dataset contiene:
- 60.000 immagini totali
- 50.000 immagini di training
- 10.000 immagini di test

Ogni immagine ha dimensione: `32 × 32 pixel`  
Questo significa che le immagini sono molto piccole, con informazioni limitate rispetto a fotografie normali.

### Classi presenti
Le 10 classi del dataset sono:
- aereo
- automobile
- uccello
- gatto
- cervo
- cane
- rana
- cavallo
- nave
- camion

Alcune classi possono essere facilmente distinguibili, mentre altre possono creare confusione.  
Ad esempio:
- automobile vs camion
- cane vs gatto
- cervo vs cavallo
- uccello vs tutto (nel nostro caso)  

soprattutto quando l'immagine è rumorosa o poco definita.

### Struttura delle immagini
Le immagini sono a colori, quindi possiedono tre canali:
- Rosso (R)
- Verde (G)
- Blu (B)

La struttura di ogni immagine è quindi: `32 × 32 × 3`  
Ogni pixel contiene valori compresi tra: `0 – 255`

### Dataset bilanciato
CIFAR-10 è un dataset bilanciato, cioè ogni classe contiene circa lo stesso numero di immagini.  
Questo è utile perché evita che il modello favorisca troppo alcune classi rispetto ad altre.  

### Difficoltà del problema
Nonostante la struttura semplice, CIFAR-10 non è un dataset banale.
Le principali difficoltà sono:
- immagini molto piccole
- oggetti spesso poco definiti
- classi visivamente simili

Inoltre il modello può facilmente incorrere in overfitting, cioè imparare troppo bene il training set ma performare peggio su immagini nuove.

Per questo vengono utilizzate tecniche come:
- dropout
- batch normalization
- data augmentation

### Perché usare una CNN
Le Convolutional Neural Networks (CNN) sono particolarmente adatte per le immagini.  
Durante l'addestramento:
- i primi layer imparano pattern semplici (bordi, linee, texture)
- i layer più profondi combinano queste informazioni
- il modello riconosce forme più complesse come ali, ruote o sagome di animali

Per questo motivo una CNN è molto più efficace rispetto a una rete neurale completamente densa per questo tipo di problema.

---

## Strategia
Il progetto segue un approccio progressivo per costruire e migliorare un modello di classificazione basato su CNN.

In una prima fase viene analizzato il dataset per comprenderne le caratteristiche e le principali difficoltà.  
Successivamente i dati vengono preparati per l'addestramento, normalizzando i pixel e organizzando correttamente training set, validation set e test set.

Viene quindi costruita una baseline, ovvero un primo modello semplice che funge da punto di partenza.  
Dopo aver osservato il comportamento di questo modello, vengono introdotti miglioramenti progressivi per ridurre l'overfitting e aumentare l'accuracy.

Infine i diversi modelli ottenuti vengono confrontati tra loro per individuare quello con le prestazioni migliori, analizzando i risultati tramite grafici di training, matrice di confusione ed esempi di predizioni corrette e sbagliate.

---

## Pipeline
Il progetto segue 18 fasi principali:

1. Import librerie
    In questa fase vengono importate tutte le librerie necessarie per lo sviluppo del progetto. In particolare si utilizzano librerie per il machine learning e deep learning, come TensorFlow/Keras, oltre a strumenti per la manipolazione dei dati (NumPy) e per la visualizzazione grafica (Matplotlib e Seaborn).
    Queste librerie permettono rispettivamente di costruire e addestrare la rete neurale, gestire i dataset e visualizzare risultati e statistiche durante le varie fasi del lavoro.



2. Caricamento Dataset
    In questa fase il dataset viene caricato e suddiviso nelle due componenti principali:
    - training set, utilizzato per addestrare il modello
    - test set, utilizzato per valutarne le prestazioni  

    Vengono inoltre stampate le dimensioni degli array per verificare che il caricamento sia avvenuto correttamente.



3. Visualizzazione di prova delle immagini
    In questa fase vengono visualizzate alcune immagini campione, generalmente una per ogni classe o una piccola griglia di esempi.  
    Questa operazione permette di:
    - verificare la corretta lettura del dataset
    - osservare la qualità e la risoluzione delle immagini
    - comprendere visivamente le classi che il modello dovrà distinguere.



4. Analisi delle classi
    In questa fase si analizza la distribuzione delle classi nel dataset.  
    Viene contato il numero di immagini per ciascuna classe e spesso viene generato un grafico a barre per visualizzare tale distribuzione.  
    Questo passaggio è importante per capire se il dataset è bilanciato, cioè se tutte le classi hanno un numero simile di esempi. Dataset molto sbilanciati possono infatti influenzare negativamente l'addestramento del modello.



5. Preprocessing
    In questa fase si preparano i dati prima dell'addestramento della rete neurale.  
    Le operazioni principali includono:
    - conversione dei pixel in formato numerico float
    - normalizzazione dei valori dei pixel (tipicamente dividendo per 255)
    - eventuale codifica delle etichette tramite one-hot encoding  
    
    Questi passaggi rendono i dati più adatti al processo di training e aiutano il modello a convergere più velocemente.



6. Baseline CNN
    In questa fase viene costruito un primo modello di CNN.
    Questo modello rappresenta una baseline, cioè una versione iniziale relativamente semplice che servirà come riferimento per i miglioramenti successivi.  
    La rete include tipicamente:
    - layer convoluzionali (Conv2D)
    - layer di pooling (MaxPooling)
    - layer completamente connessi (Dense)  
    
    Lo scopo è ottenere una prima architettura funzionante da cui partire.



7. Compilazione
    In questa fase il modello viene compilato specificando:
    - funzione di perdita (loss function), che misura l'errore del modello
    - ottimizzatore, che aggiorna i pesi durante il training
    - metriche di valutazione, come l'accuracy  
    
    Questo prepara il modello all'addestramento vero e proprio.



8. Training baseline
    In questa fase la rete neurale aggiorna progressivamente i propri pesi per migliorare la capacità di classificare correttamente le immagini, e possono essere utilizzate tecniche utili per rendere il training più stabile come:
    - validation split
    - early stopping
    - riduzione del learning rate  



9. Grafici
    In questa fase, dopo l’addestramento, vengono visualizzati i grafici relativi al training del modello.  
    In particolare si osservano sia per il training set sia per il validation set:
    - andamento della accuracy
    - andamento della loss  
    
    Questi grafici permettono di individuare fenomeni come overfitting o underfitting.



10. Valutazione
    In questa fase il modello baseline viene valutato sul dataset di test per misurarne le prestazioni reali su dati mai visti durante il training.  
    Vengono quindi calcolate metriche come:
    - accuracy finale
    - loss  
    
    Questi risultati rappresentano il punto di riferimento iniziale per confrontare i miglioramenti successivi.



11. Miglioramenti: batch normalization, dropout, data augmentation
    In questa fase si introducono diverse tecniche per migliorare le prestazioni del modello:
    - Batch Normalization, che stabilizza e accelera il training
    - Dropout, che riduce l’overfitting disattivando casualmente alcuni neuroni
    - Data Augmentation, che aumenta artificialmente la varietà dei dati generando trasformazioni delle immagini (rotazioni, traslazioni, zoom, ecc.)  
    
    Queste tecniche aiutano il modello a generalizzare meglio.



12. Training e addestramento del modello migliorato
    In questa fase il nuovo modello, arricchito con le tecniche di regolarizzazione e miglioramento, viene addestrato nuovamente sul dataset.  
    Questa fase segue una procedura simile al training precedente, ma si aspetta di ottenere prestazioni migliori grazie alla nuova architettura e alle tecniche introdotte.



13. Grafici modello migliorato
    Anche per il modello migliorato vengono generati i grafici di training e validation.
    L’analisi di questi grafici permette di capire se:
    - l’accuracy è migliorata
    - l’overfitting è diminuito
    - il training è più stabile.



14. Confronto tra modelli
    In questa fase vengono confrontati i risultati ottenuti dal modello baseline e dal modello migliorato.  
    Il confronto può includere:
    - accuracy finale
    - andamento delle curve di training
    - comportamento sulle diverse classi
    
    Questo passaggio permette di valutare l’efficacia delle modifiche introdotte.



15. Matrice di confusione
    In questa fase viene raffigurata la confusion matrix, la quale mostra:
    - quante immagini di ogni classe sono state classificate correttamente
    - quali classi vengono più frequentemente confuse tra loro.  
    
    Questo tipo di analisi aiuta a comprendere meglio i punti deboli del modello.



16. Esempi di errori e successi
    In questa fase, per rendere più concreta l’analisi, vengono mostrati alcuni esempi di:
    - predizioni corrette
    - predizioni errate  
    
    Osservare direttamente le immagini aiuta a capire perché il modello commette certi errori e quali caratteristiche visive risultano più difficili da distinguere.



17. Conclusioni
    In questa fase finale vengono riassunti i risultati ottenuti durante il progetto.  
    Si discutono:
    - le prestazioni del modello
    - i miglioramenti introdotti
    - i limiti dell’approccio adottato
    - possibili sviluppi futuri.



18. Export per front-end
    In questa fase il modello addestrato viene salvato in un formato utilizzabile al di fuori del notebook.  
    Questo permette di:
    - riutilizzare il modello senza doverlo riaddestrare
    - integrarlo in un frontend o applicazione web
    - eventualmente convertirlo in formati compatibili con altri ambienti di esecuzione.
