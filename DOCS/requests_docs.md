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