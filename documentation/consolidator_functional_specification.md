# Specifiche Funzionali: Orchestrator e Consolidator

## 1. Introduzione

### 1.1. Scopo del Documento

Questo documento fornisce una descrizione funzionale dettagliata dei servizi `orchestrator` e `consolidator`. Lo scopo è spiegare la loro architettura, il loro flusso di esecuzione, le loro interazioni con altri sistemi e la logica di elaborazione dei dati.

### 1.2. Contesto

L'`orchestrator` e il `consolidator` sono componenti chiave dell'architettura di GraphRAG. Lavorano in sequenza per orchestrare il processo di estrazione della conoscenza da documenti, la loro analisi tramite graph-based machine learning e la loro persistenza in un database strutturato per l'interrogazione.

### 1.3. Panoramica Generale

Il processo è suddiviso in due fasi principali:

1.  **Orchestrazione (gestita dall'`orchestrator`):**
    *   Recupera un documento da Google Cloud Storage.
    *   Pulisce il testo del documento.
    *   Suddivide il testo pulito in chunk di frasi complete (sentence-aware chunking).
    *   Attiva un workflow che distribuisce i chunk ai `worker` per l'elaborazione parallela.

2.  **Consolidamento (gestito dal `consolidator`):**
    *   Viene attivato al termine dell'elaborazione di tutti i chunk di un documento.
    *   Recupera i risultati parziali (entità e relazioni) da Redis.
    *   Aggrega i risultati in un unico grafo di conoscenza.
    *   Esegue algoritmi di community detection e genera riassunti per ogni community.
    *   Migra i dati finali in Google Cloud Spanner.
    *   Pulisce i dati temporanei da Redis e Memgraph.

## 2. Flusso di Esecuzione dell'Orchestrator

### 2.1. Attivazione

L'`orchestrator` è una Cloud Function attivata da un evento di creazione di un nuovo file in un bucket Google Cloud Storage.

### 2.2. Logica di Elaborazione

#### 2.2.1. Caricamento e Pulizia del Documento

*   **Input:** Evento di notifica da GCS con i dettagli del file.
*   **Logica:**
    1.  Scarica il file (PDF o testo) da GCS.
    2.  Estrae il testo grezzo dal documento.
    3.  Esegue un processo di pulizia sul testo per rimuovere tag HTML, caratteri speciali, punteggiatura non necessaria e converte il testo in minuscolo.
*   **Output:** Una stringa di testo pulito.

#### 2.2.2. Sentence-Aware Chunking

*   **Input:** Il testo pulito.
*   **Logica:**
    1.  Utilizza la libreria `nltk` per suddividere il testo in un elenco di frasi.
    2.  Raggruppa le frasi in "chunk" (blocchi di testo), assicurandosi che nessuna frase venga divisa tra due chunk e che ogni chunk non superi una dimensione massima approssimativa (es. 1000 caratteri).
*   **Output:** Una lista di chunk, dove ogni chunk è un `Document` di LangChain.

#### 2.2.3. Avvio del Workflow

*   **Input:** La lista di chunk.
*   **Logica:**
    1.  Prepara un payload per il Cloud Workflow, includendo i chunk, un `batch_id` univoco e i nomi delle risorse necessarie (code e topic).
    2.  Avvia un'esecuzione del workflow, passando il payload. Il workflow si occuperà di inviare ogni chunk a una funzione `worker` per l'elaborazione.
*   **Output:** Un'esecuzione del workflow avviata.

## 3. Flusso di Esecuzione del Consolidator

### 3.1. Attivazione

Il `consolidator` viene attivato da un messaggio Pub/Sub, che viene inviato al termine dell'elaborazione di tutti i chunk da parte dei `worker`.

### 3.2. La Catena di Consolidamento (`consolidation_chain`)

Il `consolidator` utilizza una `RunnableSequence` di LangChain per orchestrare il suo flusso di lavoro.

#### 3.2.1. `decode_pubsub_message` e `fetch_from_redis`

*   **Logica:** Estrae il `batch_id` dal messaggio Pub/Sub e lo utilizza per recuperare da Redis tutti i risultati parziali (entità e relazioni) estratti dai `worker`.

#### 3.2.2. `aggregate_results`

*   **Logica:** Aggrega i risultati parziali in un unico insieme di entità e relazioni, eliminando i duplicati.

#### 3.2.3. `load_to_memgraph`

*   **Logica:** Svuota Memgraph e carica il nuovo grafo di conoscenza aggregato.

#### 3.2.4. `run_community_detection`, `generate_summaries`, `store_summaries`

*   **Logica:** Esegue la community detection, genera riassunti testuali per ogni community utilizzando un LLM e memorizza i riassunti in Memgraph.

#### 3.2.5. `migrate_to_spanner`

*   **Logica:** Migra il grafo finale e arricchito (entità, relazioni, community, riassunti) da Memgraph a Google Cloud Spanner per la persistenza a lungo termine.

#### 3.2.6. `cleanup_redis` e `cleanup_memgraph`

*   **Logica:** Elimina i dati temporanei da Redis e svuota Memgraph per preparare il sistema alla prossima esecuzione.

## 4. Trasformazioni dei Dati

### 4.1. Pulizia del Testo (Orchestrator)

Il testo viene normalizzato per migliorare la qualità dell'input per i modelli successivi. Questo include la rimozione di elementi non testuali e la standardizzazione del formato.

### 4.2. Sentence-Aware Chunking (Orchestrator)

La suddivisione del testo in chunk basati su frasi complete garantisce che il modello di estrazione della conoscenza riceva sempre un contesto semanticamente completo, migliorando l'accuratezza dei risultati.

### 4.3. Analisi del Grafo (Consolidator)

L'utilizzo di un database a grafo e di algoritmi di community detection permette di scoprire relazioni e cluster latenti nei dati che non sarebbero evidenti con un'analisi testuale tradizionale.

### 4.4. Riassunto (Consolidator)

La generazione di riassunti tramite LLM fornisce un livello di astrazione e comprensione umana sui dati estratti, rendendoli più facilmente interpretabili.

## 5. Gestione degli Errori

Sia l'`orchestrator` che il `consolidator` sono racchiusi in blocchi `try...except` per catturare e registrare eventuali errori imprevisti durante l'esecuzione.

## 6. Configurazione

Entrambe le funzioni sono configurate tramite variabili d'ambiente per definire le informazioni di connessione ai servizi esterni (Redis, Memgraph, Spanner), i nomi delle risorse GCP e altre impostazioni operative.