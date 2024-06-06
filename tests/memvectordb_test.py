import unittest
from memvectordb.collection import MemVectorDB
from json.decoder import JSONDecodeError

class TestMemVectorDB(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.client = MemVectorDB(base_url="http://127.0.0.1:8000")


    def test_01_create_collection(self):
        """Test creating a collection."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        collection = self.client.create_collection(collection_name, dimension, distance)
        expected_string = f'Collection created: "{collection_name}"'
        self.client.delete_collection(collection_name)
        self.assertIn(expected_string, collection, f"Expected string not found in collection: {expected_string}")

    def test_02_get_collection(self):
        """Test getting a collection."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        self.client.create_collection(collection_name, dimension, distance)
        inserted_data = self.client.get_collection(collection_name)
        self.client.delete_collection(collection_name)
        self.assertEqual(dimension, inserted_data["dimension"])
        self.assertEqual(distance, inserted_data["distance"])
        self.assertEqual(0, len(inserted_data['embeddings']))

    def test_03_insert_embeddings(self):
        """Test inserting embeddings into a collection."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        self.client.create_collection(collection_name, dimension, distance)
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
        self.client.insert_embeddings(
            collection_name=collection_name, 
            vector_id=embedding["id"]['unique_id'], 
            vector=embedding["vector"], 
            metadata=embedding["metadata"]
        )
        inserted_data = self.client.get_collection(collection_name)
        self.client.delete_collection(collection_name)
        self.assertEqual(dimension, inserted_data["dimension"])
        self.assertEqual(distance, inserted_data["distance"])
        self.assertEqual(1, len(inserted_data['embeddings']))

    def test_04_batch_insert_embeddings(self):
        """Test batch inserting embeddings into a collection."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        self.client.create_collection(collection_name, dimension, distance)
        embeddings = [
            {
                "id": {
                    "unique_id": "1"
                },
                "vector": [0.14, 0.316, 0.433],
                "metadata": {
                    "key1": "value1",
                    "key2": "value2"
                }
            },
            {
                "id": {
                    "unique_id": "4"
                },
                "vector": [0.27, 0.531, 0.621],
                "metadata": {
                    "key1": "value3",
                    "key2": "value4"
                }
            }
        ]
        self.client.batch_insert_embeddings(
            collection_name, 
            embeddings 
            )
        inserted_data = self.client.get_collection(collection_name)
        self.client.delete_collection(collection_name)
        self.assertEqual(dimension, inserted_data["dimension"])
        self.assertEqual(distance, inserted_data["distance"])
        self.assertEqual(2, len(inserted_data['embeddings']))

    def test_05_get_embeddings(self):
        """Test getting embeddings from a collection."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        self.client.create_collection(collection_name, dimension, distance)
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
        try:
            self.client.insert_embeddings(
                collection_name=collection_name, 
                vector_id=embedding["id"]['unique_id'], 
                vector=embedding["vector"], 
                metadata=embedding["metadata"]
            )
        except JSONDecodeError as e:
            self.fail(f"Failed to parse JSON response: {e}")
        embeddings = self.client.get_embeddings(collection_name)
        self.client.delete_collection(collection_name)
        self.assertEqual(1, len(embeddings))
        self.assertEqual(embedding['id']['unique_id'], embeddings[0]['id']['unique_id'])
        self.assertEqual(embedding['metadata'], embeddings[0]['metadata'])

    def test_06_query(self):
        """Test querying similar vectors."""
        collection_name = "test_collection_name"
        distance = "cosine" 
        dimension = 3
        self.client.create_collection(collection_name, dimension, distance)
        embeddings = [
            {
                "id": {
                    "unique_id": "1"
                },
                "vector": [0.14, 0.316, 0.433],
                "metadata": {
                    "key1": "value1",
                    "key2": "value2"
                }
            },
            {
                "id": {
                    "unique_id": "4"
                },
                "vector": [0.27, 0.531, 0.621],
                "metadata": {
                    "key1": "value3",
                    "key2": "value4"
                }
            }
        ]
        
        self.client.batch_insert_embeddings(
            collection_name, 
            embeddings 
            )

        query_vector = [0.32, 0.24, 0.55] 
        similar_vectors = self.client.query(
            collection_name = collection_name,
            k = 1, 
            query_vector = query_vector
        )
        self.client.delete_collection(collection_name)
        self.assertEqual(1, len(similar_vectors))
        self.assertIsNotNone(similar_vectors, "The result should not be None")

    @classmethod
    def sort_test_methods(cls, testCaseClass, testCaseNames):
        """
        Sort test methods for better readability.
        """
        return sorted(testCaseNames)

if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = TestMemVectorDB.sort_test_methods
    unittest.main()
