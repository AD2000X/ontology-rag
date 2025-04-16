# src/semantic_retriever.py

# Import type hints for better readability and static checking
# Import numpy for potential numerical operations (not directly used here)
# Import OpenAI's embedding model from LangChain's community module
# Import FAISS vector store for fast similarity search
# Import Document schema from LangChain for uniform document handling
# Import the custom OntologyManager that provides semantic context and entity paths
# Define a class to perform semantic retrieval, combining vector similarity and ontology knowledge
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.ontology_manager import OntologyManager

class SemanticRetriever:
    """
    Enhanced retrieval system that combines vector search with ontology awareness.
    """

    # Initialization vector storage
    def __init__(
        self, 
        ontology_manager: OntologyManager,         # required: ontology manager instance
        embeddings_model = None,                   # optional: embedding model (defaults to OpenAIEmbeddings)
        text_chunks: Optional[List[str]] = None    # optional: additional text to embed
    ):
        """
        Initialize the semantic retriever.
        
        Args:
            ontology_manager: The ontology manager instance
            embeddings_model: The embeddings model to use (defaults to OpenAIEmbeddings)
            text_chunks: Optional list of text chunks to add to the vector store
        """
        self.ontology_manager = ontology_manager                    # Store ontology manager
        self.embeddings = embeddings_model or OpenAIEmbeddings()    # Use the provided embedding model or default to OpenAIEmbeddings
        
        # Create a vector store with the text representation of the ontology
        ontology_text = ontology_manager.get_text_representation()    # Get a text version of the ontology structure for embedding
        self.ontology_chunks = self._split_text(ontology_text)        # Split the ontology text into chunks (fixed size) for vector indexing
        
        # If additional text is given, merge it with ontology chunks
        if text_chunks:
            self.text_chunks = text_chunks
            all_chunks = self.ontology_chunks + text_chunks
        else:
            self.text_chunks = []
            all_chunks = self.ontology_chunks
        
        # Convert each chunk into a Document object with metadata for FAISS
        documents = [Document(page_content=chunk, metadata={"source": "ontology" if i < len(self.ontology_chunks) else "text"}) 
                    for i, chunk in enumerate(all_chunks)]
        
        # Create a FAISS vector store from the documents
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    # Private helper to split long text into overlapping chunks
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks for embedding."""
        chunks = []
        text_length = len(text)

        # Sliding window approach with overlap
        for i in range(0, text_length, chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 50:  # Skip very small chunks
                continue
            chunks.append(chunk)
            
        return chunks

    # Core function: Hybrid search method, combining vector retrieval and ontology context
    def retrieve(self, query: str, k: int = 4, include_ontology_context: bool = True) -> List[Document]:
        """
        Retrieve relevant documents using a hybrid approach.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            include_ontology_context: Whether to include additional ontology context
            
        Returns:
            A list of retrieved documents
        """
        # Get semantic context from the ontology
        if include_ontology_context:
            ontology_context = self.ontology_manager.get_semantic_context(query)
        else:
            ontology_context = []
        
        # Perform vector similarity search, FAISS
        vector_results = self.vector_store.similarity_search(query, k=k)
        
        # Combine results
        combined_results = vector_results
        
        # Append ontology-derived context as separate documents
        for i, context in enumerate(ontology_context):
            combined_results.append(Document(
                page_content=context,
                metadata={"source": "ontology_context", "context_id": i}
            ))
        
        return combined_results

    # Multi-hop path retrieval, contains semantic paths between entities
    def retrieve_with_paths(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Enhanced retrieval that includes semantic paths between entities.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            A dictionary containing retrieved documents and semantic paths
        """
        # Basic retrieval, get standard documents using hybrid retrieval
        basic_results = self.retrieve(query, k)
        
        # Extract potential entities from the query (simplified approach)
        # A more sophisticated approach would use NER or entity linking
        entity_types = ["Product", "Department", "Employee", "Manager", "Customer", "Feedback"]
        query_words = query.lower().split()
        
        potential_entities = []
        for entity_type in entity_types:
            if entity_type.lower() in query_words:
                # Get instances of this type
                instances = self.ontology_manager.get_instances_of_class(entity_type)
                if instances:
                    # Just take the first few for demonstration
                    potential_entities.extend(instances[:2])
        
        # Feature Search, try finding semantic paths between pairs of entities
        paths = []
        if len(potential_entities) >= 2:
            for i in range(len(potential_entities)):
                for j in range(i+1, len(potential_entities)):
                    source = potential_entities[i]
                    target = potential_entities[j]
                    
                    # Find paths between these entities, ask ontology manager to find paths up to 3 steps
                    entity_paths = self.ontology_manager.find_paths(source, target, max_length=3)
                    
                    if entity_paths:
                        for path in entity_paths:
                            # Convert path to text
                            path_text = self._path_to_text(path)
                            paths.append({
                                "source": source,
                                "target": target,
                                "path": path,
                                "text": path_text
                            })
        
        # Wrap each semantic path in a Document
        path_documents = []
        for i, path_info in enumerate(paths):
            path_documents.append(Document(
                page_content=path_info["text"],
                metadata={
                    "source": "semantic_path",
                    "path_id": i,
                    "source_entity": path_info["source"],
                    "target_entity": path_info["target"]
                }
            ))

        # Return both normal documents and enriched path data
        return {
            "documents": basic_results + path_documents,
            "paths": paths
        }

    # Helper to convert a list of RDF-style triples into a readable string
    def _path_to_text(self, path: List[Dict]) -> str:
        """Convert a path to a text description."""
        if not path:
            return ""
        
        text_parts = []
        for edge in path:
            source = edge["source"]
            target = edge["target"]
            relation = edge["type"]
            
            # Get entity information
            source_info = self.ontology_manager.get_entity_info(source)
            target_info = self.ontology_manager.get_entity_info(target)
            
            # Get names if available
            source_name = source
            if "properties" in source_info and "name" in source_info["properties"]:
                source_name = source_info["properties"]["name"]
            
            target_name = target
            if "properties" in target_info and "name" in target_info["properties"]:
                target_name = target_info["properties"]["name"]
            
            # Describe the relationship
            text_parts.append(f"{source_name} {relation} {target_name}")
        
        return " -> ".join(text_parts)

    # Search entities by class and a property value (like SQL WHERE)
    def search_by_property(self, class_type: str, property_name: str, property_value: str) -> List[Document]:
        """
        Search for instances of a class with a specific property value.
        
        Args:
            class_type: The class to search in
            property_name: The property name to match
            property_value: The property value to match
            
        Returns:
            A list of matched entities as documents
        """
        instances = self.ontology_manager.get_instances_of_class(class_type)
        
        results = []
        for instance_id in instances:
            entity_info = self.ontology_manager.get_entity_info(instance_id)
            # Check if the property exists and matches
            if "properties" in entity_info:
                properties = entity_info["properties"]
                if property_name in properties:
                    # Simple string matching (could be enhanced with fuzzy matching)
                    if str(properties[property_name]).lower() == property_value.lower():
                        # Convert to document
                        doc_content = f"Instance: {instance_id}\n"
                        doc_content += f"Type: {class_type}\n"
                        doc_content += "Properties:\n"
                        
                        for prop_name, prop_value in properties.items():
                            doc_content += f"- {prop_name}: {prop_value}\n"
                        
                        results.append(Document(
                            page_content=doc_content,
                            metadata={
                                "source": "property_search",
                                "instance_id": instance_id,
                                "class_type": class_type
                            }
                        ))
        
        return results
