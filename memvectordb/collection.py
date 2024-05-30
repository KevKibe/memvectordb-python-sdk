import requests
from typing import Dict, Any, List, Optional


class MemVectorDB:
    def __init__(
        self, 
        base_url
    ) -> None:
        self.base_url = base_url
        pass

    def create_collection(
        self,
        collection_name: str, 
        dimension: int, 
        distance: str
    ) -> str:
        """
        Create a collection with the specified parameters.

        Args:
            collection_name (str): The name of the collection.
            dimension (int): The dimension of the vectors in the collection.
            distance (str): The distance metric to use for similarity search.

        Returns:
            dict: Response from the server.
        """
        payload = {
            "collection_name": collection_name,
            "dimension": dimension,
            "distance": distance
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/create_collection"
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            status = response.json()['status']
            if status == "Error: UniqueViolation":
                return "Error: Collection with name '{}' already exists".format(collection_name)
            else:
                return status
        else:
            response.raise_for_status()

    def get_collection(
        self,
        collection_name: str
    ) -> str:
        """
        Retrieve information about a collection.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            dict: Information about the collection.
        """
        payload = {
            "collection_name": collection_name
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/get_collection"
        response = requests.get(url, json=payload, headers=headers)

        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            return response_data
        
    def delete_collection(
        self,
        collection_name: str
    ) -> str:
        """
        Delete an existing collection.

        Args:
            collection_name (str): The name of the collection to delete.

        Returns:
            dict: Confirmation statement.
        """
        payload = {
            "collection_name": collection_name
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/delete_collection"
        response = requests.delete(url, json=payload, headers=headers)

        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            return response_data

    def insert_embeddings(
        self, 
        collection_name: str, 
        vector_id: int,
        vector: List[float],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        This method inserts a single embedding into a specified collection.

        Args:
            collection_name (str): The name of the collection to insert the embedding into.
            vector_id (int): The unique identifier for the vector.
            vector (List[float]): The vector to be inserted.
            metadata (Optional[Dict]): Additional metadata associated with the vector.

        Returns:
            str: Status of the insertion operation.
        """
        if metadata:
            metadata = {str(key): str(value) for key, value in metadata.items()}
        embedding = {
            "id": str(vector_id),
            "vector": vector,
            "metadata": metadata
        }
        payload = {
            "collection_name": collection_name,
            "embedding": embedding
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/insert_embeddings"
        response = requests.put(url, json=payload, headers=headers)
        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            return f"Failed to insert embedding: {response_data}"
        
    def batch_insert_embeddings(
        self, 
        collection_name: str, 
        embeddings: List[Dict[str, Any]]
    ) -> str:
        """
        Insert a batch of embeddings into a specified collection.

        Args:
            collection_name (str): The name of the collection to insert the embeddings into.
            embeddings (List[Dict[str, Any]]): List of dictionaries representing embeddings. 
                Each dictionary should contain keys 'id' (int), 'vector' (List[float]), 
                and optional 'metadata' (List[Dict[str, Any]]).
        example: 
                {
                    "collection_name" : "test_collection_name",
                    "embeddings" :
                            [
                            {
                                "id": "1",
                                "vector": [0.14, 0.316, 0.433],
                                "metadata": {
                                    "key1": "value1",
                                    "key2": "value2"
                                }
                            },
                            {
                                "id": "2",
                                "vector": [0.27, 0.531, 0.621],
                                "metadata": {
                                    "key1": "value3",
                                    "key2": "value4"
                                }
                            }
                        ]
                }

        Returns:
            str: Status message indicating the success of the insertion operation.
        """
        for emb in embeddings:
                if emb.get("metadata"):
                    emb["metadata"] = {str(key): str(value) for key, value in emb["metadata"].items()}
        payload = {
            "collection_name": collection_name,
            "embeddings": embeddings
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/batch_insert_embeddings"
        response = requests.put(url, json=payload, headers=headers)
        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            raise Exception(f"Failed to insert embedding: {response_data}")

    def get_embeddings(
        self, 
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve embeddings from a collection.

        Args:
            collection_name (str): The name of the collection to retrieve embeddings from.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the retrieved embeddings.
        """
        payload = {
            "collection_name": collection_name
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/get_embeddings"
        response = requests.get(url, json=payload, headers=headers)

        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            return response_data
        
    def query(
        self,
        k: int,
        collection_name: str,
        query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar embeddings from a collection based on a query vector.

        Args:
            k (int): The number of similar embeddings to retrieve.
            collection_name (str): The name of the collection to retrieve embeddings from.
            query_vector (List[float]): The query vector for similarity search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the similar embeddings.
        """
        payload = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "k": k
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/get_similarity"
        response = requests.get(url, json=payload, headers=headers)

        response_data = response.json()
        if response.status_code == 200:
            return response_data
        else:
            return response_data
