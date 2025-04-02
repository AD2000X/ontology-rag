# Deployment Guide for Ontology-Enhanced RAG System

This guide will help you deploy the Ontology-Enhanced RAG demonstration to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: You need a Hugging Face account.
2. **OpenAI API Key**: You need a valid OpenAI API key.

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains the following files and directories:

- `app.py`: Main Streamlit application
- `src/`: Directory containing all source code
- `data/`: Directory containing the ontology JSON and other data
- `.streamlit/`: Directory containing Streamlit configuration
- `static/`: Directory containing CSS and other static assets
- `requirements.txt`: List of all dependencies
- `huggingface.yml`: Hugging Face Space configuration

### 2. Set Up Hugging Face Space

1. Visit [Hugging Face](https://huggingface.co/) and log in
2. Click "New" → "Space" in the top right corner
3. Fill in the Space settings:
   - **Owner**: Select your username or organization
   - **Space name**: Choose a name for your demo, e.g., "ontology-rag-demo"
   - **License**: Choose MIT or your preferred license
   - **SDK**: Select Streamlit
   - **Space hardware**: Choose according to your needs (minimum requirement: CPU + 4GB RAM)

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

2. Copy all your files into the cloned repository
3. Add, commit, and push the changes:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

### 5. Verify Deployment

1. Visit your Space URL (in the format `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`)
2. Confirm that the application loads and runs correctly
3. Test all features

## Hardware Recommendations

For optimal performance, consider the following hardware configurations:

- **Minimal**: CPU + 4GB RAM (suitable for demos with limited users)
- **Recommended**: CPU + 16GB RAM (for better performance with knowledge graph visualizations)

## Troubleshooting

If you encounter issues:

1. **Application fails to start**: 
   - Check if the Streamlit version is compatible
   - Verify all dependencies are correctly installed
   - Check the Space logs for error messages

2. **OpenAI API errors**:
   - Confirm the API key is correctly set as a Secret
   - Verify the API key is valid and has sufficient quota

3. **Display issues**:
   - Try simplifying visualizations, as they might be memory-intensive
   - Check logs for any warnings or errors

4. **NetworkX or Visualization Issues**:
   - Ensure pygraphviz is properly installed
   - For simpler deployment, you can modify the code to use alternative layout algorithms that don't depend on Graphviz

## Deployment Optimizations

For production deployments, consider these optimizations:

1. **Resource Management**:
   - Choose appropriate hardware (CPU+RAM) to meet your application's needs
   - Consider optimizing large visualizations to reduce memory usage

2. **Performance**:
   - Implement result caching for common queries
   - Consider pre-computing common graph layouts

3. **Security**:
   - Ensure no sensitive data is stored in the codebase
   - Store all credentials using environment variables or Secrets

## Memory Optimization Tips

If you encounter memory issues with large ontologies:

1. Limit the maximum number of nodes in visualization
2. Implement pagination for large result sets
3. Use streaming responses for large text outputs
4. Optimize NetworkX operations for large graphs

## Additional Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
