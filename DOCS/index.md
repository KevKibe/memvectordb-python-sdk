# MemVectorDB-Python Client

## Getting Started

```bash
pip install memvectordb-python
```

### Using Sentence Transformers as Embedding Provider

```python
from memvectordb.vectorstore import MemVectorDBVectorStore


base_url = "vectordb-url"
embedding_provider = "sentence_transformers"  
embedding_model = "multi-qa-MiniLM-L6-cos-v1"  # Specific embedding model name 

collection_name = "collection_name"
distance = "distance-metric"        # either 'cosine', 'euclidean' or 'dot'

vector_store = MemVectorDBVectorStore(
    base_url=base_url,
    embedding_provider=embedding_provider,
    embedding_model=embedding_model,
)

vector_store.create_collection(collection_name, distance)

embedding_model_client = vector_store.initialize_embedding_model_client()
```

### Using OpenAI Embeddings as Embedding Provider

```python
base_url = "base-vectordb-url"
embedding_provider = "openai"  
embedding_model = "text-embedding-3-small"  # Specific embedding model name 

collection_name = "collection_name"
distance = "distance-metric"        # either 'cosine', 'euclidean' or 'dot'

vector_store = MemVectorDBVectorStore(
    base_url=base_url,
    embedding_provider=embedding_provider,
    embedding_model=embedding_model,
    api_key= "openai-api-key"
)

vector_store.create_collection(collection_name, distance)

embedding_model_client = vector_store.initialize_embedding_model_client()
```

### Upserting Text Embeddings into the DB
```python
# Upserting texts
text = "First text string"
vector_store.add_texts(collection_name, text, embedding_model_client)
```

### Upserting Document Embeddings into the DB

```python
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


# Upserting a text document
def load_doc(file_url):
    pages = []
    file_path = Path(file_url)

    file_extension = file_path.suffix
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_url)
        pages = loader.load_and_split()
    return pages

doc = load_doc("https://arxiv.org/pdf/1706.03762.pdf")
vector_store.add_documents(collection_name, doc, embedding_model_client, streaming=True) 
# Streaming: configures how the pages are upserted into the DB ie batch or stream.
```
### To Query Vectors.

```python

query_vector = vector_store.embed(text="What is the transformer architecture", embedding_model_client=embedding_model_client)
# example of query_vector: [0.32654, 0.24423, 0.7655] 
# ensure the dimensions match the collection's dimensions
similar_vectors = vector_store.query_collection(k = 2, collection_name= 'colllection-name', query_vector=query_vector)
similar_vectors

```

### To get all embeddings in a collection

```python
vector_store.get_embeddings(collection_name)
```

### To Delete Collection

```python
vector_store.delete_collection(collection_name)
```

