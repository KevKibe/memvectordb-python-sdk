import unittest
from memvectordb.vectorstore import MemVectorDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

class TestMemVectorDBVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        base_url= "http://127.0.0.1:8000"
        self.collection_name = "test_collection"
        self.distance = "cosine"
        self.embedding_model = "multi-qa-MiniLM-L6-cos-v1"
        self.embedding_provider = "sentence_transformers"
        self.client = MemVectorDBVectorStore(base_url, self.embedding_provider, self.embedding_model)
    
    def dimensions(self):
        if self.embedding_model=="text-embedding-3-large":
            dimension = 1024
        elif self.embedding_model=="text-embedding-3-small":
            dimension = 1536
        elif self.embedding_model=="multi-qa-MiniLM-L6-cos-v1":
            dimension = 384
        return dimension
    
    def test_01_create_collection(self):
        collection = self.client.create_collection(self.collection_name, self.distance)
        expected_string = f'Collection created: "{self.collection_name}"'
        self.client.delete_collection(self.collection_name)
        self.assertIn(expected_string, collection, f"Expected string not found in collection: {expected_string}")
    
    def test_02_get_collection(self):
        """Test getting a collection."""
        self.client.create_collection(self.collection_name, self.distance)
        inserted_data = self.client.get_collection(self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.assertEqual(self.dimensions(), inserted_data["dimension"])
        self.assertEqual(self.distance, inserted_data["distance"])
        self.assertEqual(0, len(inserted_data['embeddings']))
    
    def test_03_add_texts(self):
        self.client.create_collection(self.collection_name, self.distance)
        client = self.client.initialize_embedding_model_client()
        texts = ["First text string", "Second text string", "Third text string"]
        for text in texts:
            self.client.add_texts(self.collection_name, text=text, embedding_model_client=client)
        inserted_data = self.client.get_collection(self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.assertEqual(self.dimensions(), inserted_data["dimension"])
        self.assertEqual(self.distance, inserted_data["distance"])
        self.assertEqual(3, len(inserted_data['embeddings']))
        print(inserted_data['embeddings'])
    
    def test_04_add_documents_streaming(self):
        self.client.create_collection(self.collection_name, self.distance)
        def load_doc(file_url):
            pages = []
            file_path = Path(file_url)

            file_extension = file_path.suffix
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_url)
                pages = loader.load_and_split()
            return pages
        doc = load_doc("https://arxiv.org/pdf/1706.03762.pdf")
        client = self.client.initialize_embedding_model_client()
        self.client.add_documents(self.collection_name, doc, client, streaming=True)
        inserted_data = self.client.get_collection(self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.assertEqual(self.dimensions(), inserted_data["dimension"])
        self.assertEqual(self.distance, inserted_data["distance"])
    
    def test_04_add_documents_batch(self):
        self.client.create_collection(self.collection_name, self.distance)
        def load_doc(file_url):
            pages = []
            file_path = Path(file_url)
            file_extension = file_path.suffix
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_url)
                pages = loader.load_and_split()
            return pages
        doc = load_doc("https://arxiv.org/pdf/1706.03762.pdf")
        client = self.client.initialize_embedding_model_client()
        self.client.add_documents(self.collection_name, doc, client, streaming=False)
        inserted_data = self.client.get_collection(self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.assertEqual(self.dimensions(), inserted_data["dimension"])
        self.assertEqual(self.distance, inserted_data["distance"])
    
    def test_05_query_collection(self):
        
        self.client.create_collection(self.collection_name, self.distance)
        texts = ["First text string", "Second text string", "Third text string"]
        client = self.client.initialize_embedding_model_client()
        
        for text in texts:
            self.client.add_texts(self.collection_name, text=text, embedding_model_client=client)
        client = self.client.initialize_embedding_model_client()
        query_text = "First text string"

        query_embedding = self.client.embed(query_text, client)
        similar_vector = self.client.query_collection(
            k=1, 
            collection_name=self.collection_name, 
            query_vector=query_embedding
        )
        self.client.delete_collection(self.collection_name)
        self.assertEqual(1, len(similar_vector))
        self.assertIsNotNone(similar_vector, "The result should not be None")

    @classmethod
    def sort_test_methods(cls, testCaseClass, testCaseNames):
        """
        Sort test methods for better readability.
        """
        return sorted(testCaseNames)

if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = TestMemVectorDBVectorStore.sort_test_methods
    unittest.main()