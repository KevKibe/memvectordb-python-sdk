import unittest
from memvectordb.collection import MemVectorDB

class TestMemVectorDB(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MemVectorDB(base_url="http://127.0.0.1:8000")
        self.collection_name = "test_collection_name"
        return super().setUp()
    
    def test_create_collection_method(self):
        dimension = 3
        distance = "cosine" 
        collection = self.client.create_collection(self.collection_name, dimension, distance)
        expected_string = f"Successfully created collection: Collection {{ dimension: {dimension}, distance: {distance}, embeddings: [] }}"
        self.assertIn(expected_string, collection, f"Expected string not found in collection: {expected_string}")
