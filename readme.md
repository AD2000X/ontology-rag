# 🧠 Enhanced Ontology-Powered RAG Demo

An advanced Retrieval-Augmented Generation demo integrating:
- **Structured Ontologies**: Formal knowledge representation with classes, properties, and relationships
- **Knowledge Graph**: Visualize and navigate entity relationships
- **Semantic Retrieval**: Enhanced retrieval that combines vector search with knowledge structure
- **Multi-hop reasoning**: Find connections between concepts through semantic paths
- **Interactive UI**: Explore the knowledge model and compare different RAG approaches

## ✨ Key Features

- **Ontology-Aware RAG**: Leverages structured knowledge to enhance retrieval and reasoning
- **Knowledge Graph Visualization**: Interactive graph exploration of enterprise concepts
- **Semantic Path Finding**: Discover and visualize connections between entities
- **Comparative Analysis**: See the difference between traditional and ontology-enhanced RAG
- **Reasoning Trace**: Understand how the system derives answers from the knowledge model

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🐳 Docker

```bash
# Build the Docker image
docker build -t ontology-rag-demo .

# Run the container
docker run -p 7860:7860 ontology-rag-demo
```

## 📂 Input Data

The system uses two data files:
- `data/enterprise_ontology.json`: Formal ontology representation with structured knowledge
- `data/enterprise_ontology.txt`: Plain text version for comparison

## 🔐 Secrets

Set your OPENAI_API_KEY in Streamlit secrets or environment:
```bash
export OPENAI_API_KEY=your_key_here
```

## 🧪 Example Questions

Try asking questions like:
- "How are Products and Departments related?"
- "What influence do Customers have on Products?"
- "Who is responsible for managing Employee performance?"
- "How does feedback flow through the organization?"
- "What happens during a Product's lifecycle?"

## 📘 Project Structure

```
semantic-rag-demo/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker configuration
├── data/
│   ├── enterprise_ontology.json  # Formal ontology representation
│   └── enterprise_ontology.txt   # Original text ontology
└── src/
    ├── ontology_manager.py    # Ontology handling and querying
    ├── semantic_retriever.py  # Enhanced retrieval with ontology awareness
    ├── knowledge_graph.py     # Knowledge graph construction and visualization
    └── visualization.py       # UI components for visualization
```

## 📊 Comparing RAG Approaches

This demo features a comparative analysis between:
1. **Traditional RAG**: Using only vector similarity for retrieval
2. **Ontology-Enhanced RAG**: Combining vector search with knowledge structure

The comparison highlights how structured knowledge can lead to more comprehensive and accurate answers, especially for complex queries involving multiple concepts and relationships.

## 🧩 Customization

You can extend this demo by:
- Adding more detailed ontology data
- Implementing more sophisticated entity recognition
- Enhancing the reasoning capabilities
- Adding additional data sources


# Deployment Instructions for Enhanced Ontology-RAG Project

This guide provides step-by-step instructions for pushing your enhanced ontology-RAG project to GitHub and deploying it on Hugging Face Spaces using Docker.

## Files Overview

The enhanced ontology-RAG project includes the following files:

1. **Main Application**:
   - `app.py` - The main Streamlit application

2. **Source Code**:
   - `src/__init__.py` - Package initialization
   - `src/ontology_manager.py` - Ontology handling and querying
   - `src/semantic_retriever.py` - Enhanced retrieval with ontology awareness
   - `src/knowledge_graph.py` - Knowledge graph construction and visualization
   - `src/visualization.py` - UI components for visualization

3. **Data**:
   - `data/enterprise_ontology.json` - Formal ontology representation
   - `data/enterprise_ontology.txt` - Original text ontology (keep your existing file)

4. **Configuration**:
   - `requirements.txt` - Updated dependencies
   - `Dockerfile` - Updated for Docker deployment
   - `README.md` - Updated project documentation
   - `huggingface.yml` - Configuration for Hugging Face Spaces
   - `.streamlit/config.toml` - Streamlit configuration (keep your existing file)
   - `.streamlit/secrets.toml` - API key storage (keep your existing file)

## Pushing to GitHub Using VS Code Terminal

1. **Clone your current repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/semantic-rag-demo.git
   cd semantic-rag-demo
   ```

2. **Create a new branch** for your enhanced implementation:
   ```bash
   git checkout -b ontology-enhanced-rag
   ```

3. **Create the necessary directory structure**:
   ```bash
   mkdir -p src data
   ```

4. **Add all the new files** to their respective locations as provided in the artifacts.

5. **Stage the changes**:
   ```bash
   git add .
   ```

6. **Commit the changes**:
   ```bash
   git commit -m "Add enhanced ontology-RAG implementation"
   ```

7. **Push to GitHub**:
   ```bash
   git push -u origin ontology-enhanced-rag
   ```

## Deploying to Hugging Face Spaces

### Option 1: Using the Hugging Face CLI (as configured in your workflow)

1. **Make sure your GitHub repository is up to date** with all the changes.

2. **Update your GitHub Actions workflow** (`.github/workflows/deploy.yml`) to deploy from your new branch:
   ```yaml
   name: Deploy to Hugging Face + Vercel

   on:
     push:
       branches: [ontology-enhanced-rag]  # Updated to use the new branch

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - name: Checkout
           uses: actions/checkout@v3
         - name: Push to Hugging Face
           uses: huggingface/huggingface-cli-action@v0.1.1
           with:
             api-token: ${{ secrets.HF_TOKEN }}
             repo-type: space
             repo-name: AD2000X/semantic-rag-demo  # Or choose a new name for the enhanced version
   ```

3. **Push the workflow file**:
   ```bash
   git add .github/workflows/deploy.yml
   git commit -m "Update workflow to deploy from enhanced branch"
   git push
   ```

4. The GitHub Action will automatically deploy your project to Hugging Face Spaces.

### Option 2: Manual Deployment to Hugging Face Spaces

1. **Create a new Space** on Hugging Face:
   - Go to [Hugging Face](https://huggingface.co/) and sign in
   - Click on your profile picture and select "New Space"
   - Choose "Docker" as the Space SDK
   - Give it a name (e.g., "ontology-rag-demo")
   - Set visibility (public or private)

2. **Push your repository to the Hugging Face Space**:
   ```bash
   # Add Hugging Face Space as a remote
   git remote add space https://huggingface.co/spaces/your-username/ontology-rag-demo
   
   # Push to the space
   git push space ontology-enhanced-rag:main
   ```

3. Hugging Face will build and deploy your Docker container automatically.

## Important Notes

1. **API Keys**: Make sure your OpenAI API key is properly set in the Hugging Face Space:
   - Go to your Space settings
   - Add a new secret with name `OPENAI_API_KEY` and your API key as the value

2. **Resource Requirements**: The enhanced version uses more libraries and requires more computational resources, so make sure your Hugging Face Space has sufficient resources allocated.

3. **Private Spaces**: If you're using OpenAI API keys, consider making your Space private to prevent unauthorized API usage.

4. **Debugging Deployment Issues**:
   - Check the build logs in the Hugging Face Space UI
   - Test the Docker build locally before pushing:
     ```bash
     docker build -t ontology-rag-demo .
     docker run -p 7860:7860 ontology-rag-demo
     ```

5. **Merging to Main**: Once everything is working as expected on your branch, you can merge it to main:
   ```bash
   git checkout main
   git merge ontology-enhanced-rag
   git push origin main
   ```

Your enhanced ontology-RAG project should now be successfully deployed on Hugging Face Spaces, showcasing your understanding of ontologies and their integration with RAG systems.