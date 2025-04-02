# Deployment Guide for Ontology-Enhanced RAG System

This guide provides detailed instructions for deploying the Ontology-Enhanced RAG demonstration to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: You need a Hugging Face account to create and manage Spaces.
2. **OpenAI API Key**: You need a valid OpenAI API key with sufficient quota for embeddings and completions.

## Repository Structure Overview

Before deployment, ensure your repository contains all necessary files:

```
ontology-rag/
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── data/
│   ├── enterprise_ontology.json  # Enterprise ontology data (structured)
│   └── enterprise_ontology.txt   # Simplified text representation of ontology
├── src/
│   ├── __init__.py
│   ├── knowledge_graph.py      # Knowledge graph processing
│   ├── ontology_manager.py     # Ontology management
│   ├── semantic_retriever.py   # Semantic retrieval
│   └── visualization.py        # Visualization functions
├── static/
│   └── css/
│       └── styles.css          # Custom styles
├── app.py                      # Main application
├── requirements.txt            # Dependencies list
├── .gitattributes              # Git LFS configuration
├── huggingface.yml             # Hugging Face Space configuration
├── DEPLOYMENT_GUIDE.md         # This file
└── README.md                   # Project overview
```

## Detailed Deployment Steps

### 1. Prepare Your Repository

Ensure your `requirements.txt` includes all necessary dependencies:

```
streamlit>=1.44.0
openai>=1.2.0
langchain>=0.1.13
langchain-community>=0.0.21
langchain-openai>=0.0.5
faiss-cpu>=1.7.4
networkx>=3.1
pyvis>=0.3.2
plotly>=5.15.0
pandas>=2.0.0
matplotlib>=3.7.1
numpy>=1.24.3
pydantic>=1.10.8
```

### 2. Set Up Hugging Face Space

1. Visit [Hugging Face](https://huggingface.co/) and log in
2. Click "New" → "Space" in the top right corner
3. Fill in the Space settings:
   - **Owner**: Select your username or organization
   - **Space name**: Choose a name for your demo, e.g., "ontology-rag-demo"
   - **License**: Choose MIT or your preferred license
   - **SDK**: Select Streamlit
   - **Space hardware**: CPU + 16GB RAM is recommended for good performance with knowledge graph visualizations

4. Click "Create Space"

### 3. Configure Space Secrets

You need to add your OpenAI API key as a Secret:

1. In your Space page, go to the "Settings" tab
2. Scroll down to the "Repository secrets" section
3. Click "New secret"
4. Add the following secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key
5. Click "Add secret"

The application is configured to access this secret using `st.secrets["OPENAI_API_KEY"]` in the code.

### 4. Upload Your Code

There are two ways to upload your code:

#### Option A: Upload via Web Interface

1. In your Space page, go to the "Files" tab
2. Use the upload button to upload all necessary files and directories
3. Ensure you maintain the correct directory structure

#### Option B: Upload via Git (Recommended)

1. Clone your Space repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

2. Copy all your files into the cloned repository, maintaining the directory structure
3. Add, commit, and push the changes:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

For large files like the ontology JSON, you might need to use Git LFS. The `.gitattributes` file in the repository is already configured for this.

### 5. Understanding the Application Architecture

This will help when troubleshooting deployment issues:

1. **Initialization Flow**:
   - The application initializes the OntologyManager first with the JSON data
   - Then initializes KnowledgeGraph using the OntologyManager
   - Finally initializes SemanticRetriever using the OntologyManager

2. **Key Components**:
   - **OntologyManager**: Handles loading and querying the ontology data
   - **KnowledgeGraph**: Manages graph construction and visualization
   - **SemanticRetriever**: Combines vector search with ontology-aware retrieval

3. **Visualization Pipeline**:
   - Graph visualizations use NetworkX for graph structure
   - PyVis for interactive HTML visualizations
   - Plotly for charts and data visualizations
   - These are rendered in Streamlit using components.html

### 6. Verify Deployment

1. Visit your Space URL (in the format `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`)
2. The application should load and display the main interface
3. Test the following critical features to ensure correct deployment:
   - RAG comparison demonstration
   - Knowledge graph visualization
   - Entity exploration

### 7. Hardware Recommendations

Based on the application's resource usage:

- **CPU + 4GB RAM**: Minimum requirement, may experience slowness with large graphs
- **CPU + 8GB RAM**: Adequate for most demos with moderate usage
- **CPU + 16GB RAM**: Recommended for smooth experience with knowledge graph visualizations
- **CPU + 32GB RAM**: For production use or large ontologies

### 8. Memory Optimization

If you encounter memory issues during deployment:

1. **Ontology Loading Optimization**:
   - The application loads the entire ontology into memory during initialization
   - For very large ontologies, consider modifying `ontology_manager.py` to implement lazy loading or pagination

2. **Graph Visualization Optimization**:
   - In `knowledge_graph.py`, adjust the `build_visualization_graph` method to limit the number of nodes:
   ```python
   # Add a parameter to limit max nodes
   max_nodes = 100  # Adjust based on memory constraints
   if len(subgraph.nodes) > max_nodes:
       # Implement node prioritization strategy
   ```

3. **Vector Store Management**:
   - The FAISS vector store can consume significant memory with large text corpora
   - Consider implementing chunking strategies in `semantic_retriever.py`

4. **Result Caching**:
   - Implement Streamlit's caching mechanisms to avoid redundant computations:
   ```python
   @st.cache_data
   def compute_expensive_operation():
       # Your computation here
   ```

5. **Streaming Responses**:
   - For large text outputs, modify the OpenAI API call to use streaming responses

### 9. Troubleshooting Common Issues

1. **Application fails to start**:
   - Check Hugging Face Space logs for specific error messages
   - Verify all dependencies are correctly installed by checking `requirements.txt`
   - Ensure your OpenAI API key is set correctly as a Secret

2. **Visualization rendering issues**:
   - If graph visualizations don't appear, check browser console for JavaScript errors
   - Try reducing the complexity of visualizations by limiting node count
   - Verify that PyVis is correctly installed

3. **Memory errors**:
   - Increase your Space's hardware tier if available
   - Implement the memory optimization techniques described above
   - Consider splitting large ontologies into smaller components

4. **Slow performance**:
   - RAG processes can be compute-intensive; consider pre-computing embeddings
   - Optimize the knowledge graph traversal in `knowledge_graph.py`
   - Use caching for frequently accessed data

5. **"Missing secrets" error**:
   - Ensure the OpenAI API key is correctly set in Space secrets
   - Verify that the name matches exactly: `OPENAI_API_KEY`

### 10. Customization Options

To customize the deployment for your specific needs:

1. **Custom Ontology**:
   - Replace `data/enterprise_ontology.json` with your own ontology following the same structure
   - Update `data/enterprise_ontology.txt` with a plain text summary of your ontology

2. **Embedding Models**:
   - You can modify `semantic_retriever.py` to use different embedding models

3. **UI Customization**:
   - Adjust theme colors in `.streamlit/config.toml`
   - Modify page layout and components in `app.py`

4. **LLM Configuration**:
   - Change the model used for generation by updating the model parameter in `app.py`:
   ```python
   enhanced_response = llm.chat.completions.create(
       model="gpt-4",  # Change to your preferred model
       messages=enhanced_messages
   )
   ```

### 11. Advanced Deployment Options

For production deployments, consider these advanced options:

1. **API Backend**:
   - Implement a FastAPI backend for your RAG system for better scalability
   - Separate the computational aspects from the UI

2. **Database Integration**:
   - Add vector database support (Pinecone, Weaviate, etc.) for more efficient retrieval

3. **Authentication**:
   - Implement Streamlit authentication for protected deployments

4. **CI/CD Pipeline**:
   - Set up continuous integration for automated testing and deployment

### 12. Additional Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [PyVis Documentation](https://pyvis.readthedocs.io/en/latest/)
- [Plotly Documentation](https://plotly.com/python/)
