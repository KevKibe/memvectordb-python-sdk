from openai import OpenAI
from .collection import MemVectorDB
import uuid
from tqdm import tqdm
from typing import List, Any, Dict

class MemVectorDBVectorStore:
    def __init__(
        self,
        base_url: str,
        embedding_model: str,
        openai_api_key: str
        ) -> None:
        self.client = MemVectorDB(
            base_url=base_url
            )
        self.embedding_model=embedding_model
        self.openai_client = OpenAI(api_key=openai_api_key)
        pass
    
    def create_collection(
        self,
        collection_name: str,
        distance: str,
        ) -> str:
        """
        Creates a collection in the vector store.

        Args:
            collection_name (str): The name of the collection.
            distance (str): The distance metric to be used.

        Returns:
            str: Status message indicating the success of the operation.
        """
        if self.embedding_model=="text-embedding-3-large":
            dimension = 1024
        else:
            dimension = 1536
        return self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance=distance
        )
    def get_collection(
        self,
        collection_name: str
    ) -> str:
        """
        Retrieves details of a specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            str: Details of the collection.
        """
        return self.client.get_collection(collection_name=collection_name)
    
    def add_texts(
        self,
        collection_name: str,
        text: str,
        ) -> str:
        """
        Adds a single text to the specified collection.

        Args:
            collection_name (str): The name of the collection.
            text (str): The text to be added.

        Returns:
            str: Status message indicating the success of the operation.
        """
        vector_id = str(uuid.uuid4())
        embeddings = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
            )
        return self.client.insert_embeddings(
            collection_name=collection_name,
            vector_id=vector_id,
            vector=embeddings.data[0].embedding,
            metadata={"text": text}
        )
    
    def add_documents(
        self,
        collection_name: str,
        documents: list,
        streaming: bool
    ) -> str:
        """
        Adds multiple documents to the specified collection.

        Args:
            collection_name (str): The name of the collection.
            documents (list): The documents to be added.
            streaming (bool): Whether to stream the documents or not.

        Returns:
            str: Status message indicating the success of the operation.
        """
        try:
            if streaming:
                for page in tqdm(documents, desc="Adding documents"):
                    try:
                        embeddings = self.openai_client.embeddings.create(
                            input=page.page_content,
                            model=self.embedding_model
                        )
                        vector_id = str(uuid.uuid4())
                        metadata = page.metadata
                        metadata["text"] = page.page_content

                        self.client.insert_embeddings(
                            collection_name=collection_name,
                            vector_id=vector_id,
                            vector=embeddings.data[0].embedding,
                            metadata=metadata
                        )
                    except Exception as e:
                        print(f"An error occurred while adding a document: {e}")
                result = "Streaming insertion completed."
            else:
                doc_embeddings = []
                for page in documents:
                    embeddings = self.openai_client.embeddings.create(
                        input=page.page_content,
                        model=self.embedding_model
                    )
                    metadata = page.metadata
                    metadata["text"] = page.page_content

                    doc_embeddings.append({
                        "id": str(uuid.uuid4()),
                        "vector": embeddings.data[0].embedding,
                        "metadata": metadata
                    })

                result = self.client.batch_insert_embeddings(
                    collection_name=collection_name,
                    embeddings=doc_embeddings
                )
            return result
        except Exception as e:
            print(f"An error occurred during document processing: {e}")
            return "Error occurred during document processing."
    
    def query_collection(
        self,
        k: int,
        collection_name: str,
        query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Queries the collection for the nearest vectors.

        Args:
            k (int): The number of nearest vectors to return.
            collection_name (str): The name of the collection.
            query_vector (List[float]): The query vector.

        Returns:
            List[Dict[str, Any]]: The list of nearest vectors.
        """
        return self.client.query(
            k=k,
            collection_name=collection_name,
            query_vector=query_vector
        )

    def delete_collection(
        self,
        collection_name: str
    ) -> str:
        """
        Deletes a specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            str: Status message indicating the success of the operation.
        """
        return self.client.delete_collection(
            collection_name=collection_name
        )

    def get_embeddings(
        self,
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all embeddings from a specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            List[Dict[str, Any]]: The list of all embeddings.
        """
        return self.client.get_embeddings(
            collection_name=collection_name
        )
