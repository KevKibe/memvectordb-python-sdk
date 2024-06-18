# memvectorDB-python client

## Getting Started

```bash
pip install memvectordb-python
```
## VectorStore
```python
from memvectordb.vectorstore import MemVectorDBVectorStore
```

### Using Sentence Transformers

```python
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

### Using OpenAI Embeddings

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

```python
# Upserting texts
text = "First text string"
vector_store.add_texts(collection_name, text, embedding_model_client)


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
## To Query Vectors.

```python
k = "number-of-items-to query"
collection_name = "collection_name"
query_vector = "query_vector"

# example of query_vector: [0.32654, 0.24423, 0.7655] 
# ensure the dimensions match the collection's dimensions
vector_store.query(collection_name, k, query_vector)
```



## Using Requests
```bash
pip install memvectordb-python
```

### To Initialize the Client

```python
from memvectordb.collection import Collection

client = MemVectorDB(base_url = "base-url") # default http://127.0.0.1:8000
```

### To Create Collection

```python
# To create a new collection
collection_name = "collection_name"
dimension = "dimension-of-vectors-to-be-stored"
distance = "distance-metric" # either 'cosine', 'euclidean' or 'dot'
collection = client.create_collection(collection_name, dimension, distance)

```

### To Get Collection

```python
collection_name = "collection_name"
collection = client.get_collection(collection_name)
```

### To Delete collection

```python
collection_name = "collection_name"
collection = client.delete_collection(collection_name)
```
### To Insert Vectors(streaming)
```python

collection_name = "collection_name"
embedding = {
    "id": {
        "unique_id": "1"
    },
    "vector": [0.14, 0.316, 0.433],
    "metadata": {
        "key1": "value1",
        "key2": "value2"
    }
}
client.insert_embeddings(
    collection_name=collection_name, 
    vector_id=embedding["id"]['unique_id'], 
    vector=embedding["vector"], 
    metadata=embedding["metadata"]
)
```

### To Insert Vectors(batch)

```python

collection_name = "collection_name"
embeddings = [
    {
        "id": {
            "unique_id": "0"
        },
        "vector": [0.14, 0.316, 0.433],
        "metadata": {
            "key1": "value1",
            "key2": "value2"
        }
    },
    {
        "id": {
            "unique_id": "1"
        },
        "vector": [0.27, 0.531, 0.621],
        "metadata": {
            "key1": "value3",
            "key2": "value4"
        }
    },
    {
        "id": {
            "unique_id": "2"
        },
        "vector": [0.27, 0.531, 0.621],
        "metadata": {
            "key1": "value3",
            "key2": "value4"
        }
    }
]

client.batch_insert_embeddings(
        collection_name=collection_name, 
        embeddings = embeddings
    )
```
## To Query Vectors.

```python
k = "number-of-items-to query"
collection_name = "collection_name"
query_vector = "query_vector"

# example of query_vector: [0.32654, 0.24423, 0.7655] 
# ensure the dimensions match the collection's dimensions
client.query(collection_name, k, query_vector)
```