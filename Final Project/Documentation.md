# IITB AI Assistant
Every LLM we have used so far be it ChatGPT, Claude, Gemini, knows a great deal about the world, but nothing at all about your world. Ask it about IIT Bombay's academic or campus life, the exact grading policy for a specific course, or what are the various tech teams and clubs running in the campus, and it will either guess, hedge, or confidently make something up. This is the exact gap Retrieval-Augmented Generation(RAG) is used for.

This AI assistant solves some of the problems like what are the various club activities taking place in the campus, what the tech teams have achieved and made our IITB proud of so far, what oppertunites students can get from these tech activities etc.

This AI assistant works on a layer implementing RAG with the API of Gemini("gemini-3.5-flash"). The speciality of this chatbot is that whenever it does not know the answer of any question it explicitly says that it don't know instaed of just beating the bush and giving self cooked answers.

## The RAG pipeline works as follows:-
### 1. Dataset
The pdfs, website links of various clubs and and tech teams are used as he dataset for this rag pipeline. For pdfs it extract text from all the non-empty pages and store them as a dictionary name loaded with new line and appended to the dictionary named doc which collect data from all the sources. In case of the websites it scraps all the texts and appends it to docs.

### 2. Chunking 
The texts from pdfs and websites stored in docs are divided into split at the '\n', each chunk is divided in a window of 700 characters so many chunks of fixed size are made from the texts.

```
def chunk_markdown(text, source): #splitting of texts about '\n'
    sections = re.split(r'\n(?=## )', text)
    return [{"text": s.strip(), "source": source} for s in sections if s.strip()]

def chunk_plain(text, source, chunk_size=800, overlap=100):#overlap of 100 char for retention of context
    chunks = []#creation of chunks
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        c = text[i:i+chunk_size].strip()
        if c:
            chunks.append({"text": c, "source": source})
    return chunks

all_chunks = [] #chunks stored in a dictionary
for doc in docs:
    if doc["type"] == "markdown":
        all_chunks.extend(chunk_markdown(doc["text"], doc["source"]))
    else:  # "web" or "pdf" — both plain text, same treatment
        all_chunks.extend(chunk_plain(doc["text"], doc["source"]))

print(f"Total chunks: {len(all_chunks)}")
```

### 3. Embedding
Importing a sentence embeddings model from hugging face, it's trained so that a query embedding and the embedding of a passage that answers it end up close together in vector space, using cosine similarity as the intended distance metric.
Runs every chunk through the model to produce a dense vector embedding for each (shape ends up [num_chunks, embedding_dim], and for this model embedding_dim = 384).
Normalizes every embedding vector in-place to unit length (L2 norm = 1). This is the key step that makes cosine similarity work with FAISS's inner-product index: once vectors are unit-normalized, the dot product between two vectors equals their cosine similarity. Without this step, IndexFlatIP would just compute raw dot products, which are sensitive to vector magnitude, not just direction.

```
embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

texts = [c["text"] for c in all_chunks]
embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1]) # creation of FAISS index 
index.add(embeddings) # Loads all the normalized embeddings into the index
print(f"Indexed {index.ntotal} chunks")
```

### 4. Retrieve
Takes the query string ad k(the number of top matching chunks to return), Embeds the query using the same embedder model used for the chunks,
Normalizes the query vector to unit length, in-place — the same step applied to the chunk embeddings earlier.
FAISS compares q_vec against every stored embedding and returns the k closest matches. scores is shape [1, k] — the similarity scores (cosine similarity, since everything's normalized, so values range roughly -1 to 1, with higher = more similar). idxs is shape [1, k] — the positions of those top-k chunks in the original embeddings/all_chunks array.

```
def retrieve(query, k=4):
    q_vec = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k)
    return [all_chunks[i] for i in idxs[0]] #idxs[0], since there is only one query 
```

### 5. Answer Generation
takes the fully-assembled prompt (retrieved chunks + the user's question, formatted together) and returns the model's answer as a string.
