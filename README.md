---
title: Ontology-Enhanced RAG System
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
---

![Ontology-RAG System](https://github.com/AD2000X/ontology-rag/blob/main/images/pipeline.jpg)

# Ontology-Enhanced RAG System

## Project Overview

This repository contains an advanced Retrieval-Augmented Generation (RAG) system that integrates structured ontologies with language models. The system demonstrates how formal ontological knowledge representation can enhance traditional vector-based retrieval methods to provide more accurate, contextually rich, and logically consistent answers to user queries.

![Ontology-RAG System](https://github.com/AD2000X/ontology-rag/blob/main/images/11.jpg)

The project implements a sophisticated architecture that combines:

- JSON-based ontology representation with classes, relationships, rules, and instances  
- Knowledge graph visualization for exploring entity relationships  
- Semantic path finding for multi-hop reasoning between concepts  
- Comparative analysis between traditional vector-based RAG and ontology-enhanced RAG  

## Technology Stack

This application uses a comprehensive tech stack:

- **Frontend**: [Streamlit](https://streamlit.io/) for the interactive web interface
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- **Graph Processing**: [NetworkX](https://networkx.org/) for graph data structures and algorithms
- **Visualization**:
  - [PyVis](https://pyvis.readthedocs.io/) for interactive network visualizations
  - [Plotly](https://plotly.com/python/) for charts and data visualizations
  - [Matplotlib](https://matplotlib.org/) for basic plotting
- **Language Models**: [OpenAI API](https://openai.com/api/) for text embeddings and completions
- **LLM Framework**: [LangChain](https://www.langchain.com/) for RAG implementation
- **Data Processing**: [Pandas](https://pandas.pydata.org/) for data manipulation

## Key Features

### 1. RAG Comparison Demo

The system provides a side-by-side comparison of traditional RAG and ontology-enhanced RAG:

![RAG Comparison Demo](https://github.com/AD2000X/ontology-rag/blob/main/images/12.jpg)
![RAG Comparison Demo](https://github.com/AD2000X/ontology-rag/blob/main/images/13.jpg)

- **Traditional RAG**: Uses only vector similarity to retrieve relevant documents
- **Ontology-Enhanced RAG**: Combines vector search with semantic relationship traversal
- **Interactive Comparison**: Enter your query and compare the responses and retrieved contexts
- **Difference Analysis**: Highlights the key advantages of ontology-enhanced retrieval

Example usage:
1. Navigate to the "RAG comparison demonstration" page
2. Enter a question like "How does customer feedback influence product development?"
3. Compare the responses from both approaches and the sources of information they used

### 2. Knowledge Graph Visualization

Interactive visualization of the ontology as a knowledge graph:

![Knowledge Graph](https://github.com/AD2000X/ontology-rag/blob/main/images/21.jpg)
![Knowledge Graph](https://github.com/AD2000X/ontology-rag/blob/main/images/22.jpg)

- **Multiple Layout Algorithms**: Force-directed, hierarchical, radial, and circular layouts
- **Entity Focus**: Zoom in on specific entities and their relationships
- **Customizable Visualization**: Control which elements (classes, instances, properties) are displayed
- **Graph Statistics**: View metrics about the graph structure, including centrality and connectivity

Example usage:
1. Navigate to the "Knowledge graph visualization" page
2. Select layout algorithm (e.g., Force-Directed)
3. Use the "Focus on Entity" expander to select a specific entity
4. Adjust the "Max Relationship Distance" slider to control the neighborhood size

### 3. Ontology Structure Analysis

Tools for exploring and understanding the ontology's structure:

![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/31.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/32.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/33.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/34.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/35.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/36.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/37.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/38.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/39.jpg)
![Ontology Structure](https://github.com/AD2000X/ontology-rag/blob/main/images/391.jpg)


- **Class Hierarchy Visualization**: View the ontology's class inheritance structure
- **Class Statistics**: See instance distribution across classes
- **Relationship Analysis**: Analyze relationship usage patterns
- **Domain-Range Distribution**: Understand how entity types are connected through relationships

Example usage:
1. Navigate to the "Ontology structure analysis" page
2. Explore the "Class Hierarchy" tab to understand inheritance relationships
3. Check "Class Statistics" tab to see which classes have the most instances
4. View "Relationship Analysis" to understand connectivity patterns

### 4. Entity Exploration

Detailed view of individual entities and their connections:

![Entity Exploration](https://github.com/AD2000X/ontology-rag/blob/main/images/41.jpg)
![Entity Exploration](https://github.com/AD2000X/ontology-rag/blob/main/images/42.jpg)

- **Entity Information Cards**: View all properties and relationships of specific entities
- **Relationship Graphs**: Visualize how an entity connects to others
- **Neighborhood Exploration**: Browse entities at different distances from the focus entity
- **Property Details**: Examine all attributes of selected entities

Example usage:
1. Navigate to the "Entity exploration" page
2. Select an entity from the dropdown (e.g., "product1")
3. View its properties and relationships in the information card
4. Click "View this Entity in the Knowledge Graph" to see a graph visualization
5. Explore its neighborhood using the distance slider

### 5. Semantic Path Visualization

Visualization of paths between entities with meaningful relationships:

![Semantic Paths](https://github.com/AD2000X/ontology-rag/blob/main/images/51.jpg)
![Semantic Paths](https://github.com/AD2000X/ontology-rag/blob/main/images/52.jpg)
![Semantic Paths](https://github.com/AD2000X/ontology-rag/blob/main/images/53.jpg)
![Semantic Paths](https://github.com/AD2000X/ontology-rag/blob/main/images/54.jpg)
![Semantic Paths](https://github.com/AD2000X/ontology-rag/blob/main/images/55.jpg)

- **Path Discovery**: Find and visualize all paths between selected entities
- **Path Explanation**: Step-by-step breakdown of each path
- **Visual Representation**: Clear visualization of paths through the knowledge graph
- **Business Rule Connection**: Link paths to relevant business rules in the ontology

Example usage:
1. Navigate to the "Semantic path visualization" page
2. Select source entity (e.g., "customer1") and target entity (e.g., "product1")
3. Adjust maximum path length if needed
4. View discovered paths and their visualizations

### 6. Reasoning Trace Visualization

Explanation of the RAG system's reasoning process:

![Reasoning Trace](https://github.com/AD2000X/ontology-rag/blob/main/images/61.jpg)
![Reasoning Trace](https://github.com/AD2000X/ontology-rag/blob/main/images/62.jpg)

- **Query Analysis**: Entity and relationship detection in user queries
- **Information Flow**: Sankey diagrams showing how information is used
- **Reasoning Steps**: Explanation of the system's reasoning process
- **Ontological Advantages**: Highlighting how the ontology enhances answers

Example usage:
1. First run a query on the "RAG comparison demonstration" page
2. Navigate to the "Inference tracking" page
3. Explore the "Query Analysis," "Knowledge Retrieval," and "Reasoning Path" tabs

### 7. Detailed Comparative Analysis

In-depth analysis of both RAG approaches:

![Comparative Analysis](https://github.com/AD2000X/ontology-rag/blob/main/images/71.jpg)
![Comparative Analysis](https://github.com/AD2000X/ontology-rag/blob/main/images/72.jpg)
![Comparative Analysis](https://github.com/AD2000X/ontology-rag/blob/main/images/73.jpg)
![Comparative Analysis](https://github.com/AD2000X/ontology-rag/blob/main/images/74.jpg)

- **Performance Metrics**: Response time, context tokens, retrieved documents
- **Retrieval Source Comparison**: Distribution of information sources
- **Context Quality Assessment**: Evaluation of retrieved context relevance and richness
- **Use Case Recommendations**: Guidance on when to use each approach

Example usage:
1. Navigate to the "Detailed comparative analysis" page
2. Select a comparison query or enter a custom one
3. Click "Compare RAG Methods"
4. Explore all tabs to understand different aspects of the comparison

## Ontology Structure

The `data/enterprise_ontology.json` file contains a rich enterprise ontology that models organizational knowledge. Here's a breakdown of its key components:

### Classes (Entity Types)

The ontology defines a hierarchical class structure with inheritance relationships:

```
Entity (base class)
├── FinancialEntity → Budget, Revenue, Expense
├── Asset → PhysicalAsset, DigitalAsset, IntellectualProperty
├── Person → InternalPerson → Employee → Manager
├── Process → BusinessProcess, DevelopmentProcess, SupportProcess
└── Market → GeographicMarket, DemographicMarket, BusinessMarket
```

Each class has a description and a set of defined properties.

### Relationships

The ontology defines explicit relationships between entity types, including:

- `ownedBy`: Connects Product to Department
- `managedBy`: Connects Department to Manager
- `worksOn`: Connects Employee to Product
- `purchases`: Connects Customer to Product
- `provides`: Connects Customer to Feedback
- `optimizedBy`: Relates Product to Feedback

Each relationship has metadata such as domain, range, cardinality, and inverse relationship name.

### Business Rules

The ontology contains formal business rules that constrain the knowledge model:

- "Every Product must be owned by exactly one Department"
- "Every Department must be managed by exactly one Manager"
- "Critical support tickets must be assigned to Senior employees or managers"
- "Product Lifecycle stages must follow a predefined sequence"

### Instances

The ontology includes concrete instances of the defined classes, such as:

- `product1`: An "Enterprise Analytics Suite" owned by the Engineering department
- `manager1`: A director named "Jane Smith" who manages the Engineering department
- `customer1`: "Acme Corp" who has purchased product1 and provided feedback

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ontology-rag.git
   cd ontology-rag
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Option 1: Set as an environment variable:
     ```bash
     export OPENAI_API_KEY=your_api_key_here
     ```
   - Option 2: Create a `.streamlit/secrets.toml` file with:
     ```toml
     OPENAI_API_KEY = "your_api_key_here"
     ```

### Running the Application

To run the application locally:

```bash
streamlit run app.py
```

The application will start and open in your default web browser at http://localhost:8501.

### Using the Application

1. **Starting Point**: Begin with the "RAG comparison demonstration" to understand the basic difference between traditional and ontology-enhanced RAG
2. **Exploration**: Use the "Knowledge graph visualization" and "Entity exploration" to understand the ontology structure
3. **Advanced Analysis**: Try "Semantic path visualization" to see how entities connect to each other
4. **Performance Comparison**: Use "Detailed comparative analysis" to dive deeper into the differences between approaches

## Extending the System

### Creating Your Own Ontology

To use the system with your custom ontology:

1. Create a JSON file following the structure in `data/enterprise_ontology.json`
2. Your ontology should define:
   - `classes`: Entity types with properties and inheritance relationships
   - `relationships`: Connection types between entities
   - `rules`: Business rules and constraints
   - `instances`: Concrete examples of entities

3. Update the OntologyManager initialization in `app.py`:
   ```python
   ontology_manager = OntologyManager("data/your_ontology.json")
   ```

### Adding More Data Sources

To expand the system's knowledge:

1. Create a text representation of your ontology to create embeddings
2. Add additional text chunks when initializing the SemanticRetriever:
   ```python
   semantic_retriever = SemanticRetriever(
       ontology_manager=ontology_manager,
       text_chunks=your_additional_text_chunks
   )
   ```

## Use Cases

### Enterprise Knowledge Management

The ontology-enhanced RAG system helps organizations effectively organize and access their knowledge assets:

- **Knowledge Integration**: Connect information across different departments and systems
- **Relationship Discovery**: Identify non-obvious connections between business entities
- **Consistent Answers**: Ensure responses adhere to organizational policies and rules

Implementation steps:
1. Model your organization's knowledge as an ontology
2. Define key entity types (products, departments, employees, etc.)
3. Establish relationships between entities
4. Add business rules as constraints
5. Populate with representative instances

### Product Development Decision Support

Support product development decisions by connecting customer feedback, product features, and market data:

- **Feedback Analysis**: Trace how customer feedback influences product features
- **Impact Assessment**: Understand how changes affect downstream dependencies
- **Cross-Department Collaboration**: Connect marketing insights with development priorities

Implementation steps:
1. Define product, feature, and feedback entities in your ontology
2. Establish relationships between customer feedback and product components
3. Create semantic paths connecting market data to development initiatives
4. Use the RAG comparison to answer development strategy questions

### Complex Compliance Queries

Ensure recommendations comply with all applicable policies and regulations:

- **Rule-Based Reasoning**: Apply multiple constraints simultaneously
- **Audit Trails**: Show reasoning paths to demonstrate compliance
- **Policy Connections**: Link entities to relevant regulatory requirements

Implementation steps:
1. Add compliance rules to your ontology
2. Create relationships between regulated entities and relevant policies
3. Use semantic path visualization to trace compliance relationships
4. Query the system about specific compliance scenarios

### Diagnostics and Troubleshooting

Connect symptoms, causes, and solutions through multi-hop reasoning:

- **Root Cause Analysis**: Trace issues through connected systems
- **Solution Recommendation**: Find relevant fixes based on symptom patterns
- **Component Relationship Mapping**: Understand dependencies between system components

Implementation steps:
1. Model system components and their relationships
2. Define typical symptoms and their possible causes
3. Link solutions to specific problem patterns
4. Use the semantic path visualization to trace from symptoms to solutions

## Project Structure

```
ontology-rag/
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── data/
│   ├── enterprise_ontology.json  # Enterprise ontology data
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
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
└── README.md                   # This file
```

## Core Components

### OntologyManager (`src/ontology_manager.py`)

Handles loading and querying the ontology:
- Loads ontology from JSON file
- Builds a graph representation
- Provides methods to query classes, instances, and relationships
- Generates text representation for embeddings

### KnowledgeGraph (`src/knowledge_graph.py`)

Manages the knowledge graph operations:
- Builds visualization graphs
- Calculates graph statistics
- Finds paths between entities
- Generates HTML visualizations

### SemanticRetriever (`src/semantic_retriever.py`)

Implements the RAG functionality:
- Creates embeddings for ontology text
- Performs vector similarity search
- Enhances retrieval with semantic paths
- Combines different retrieval methods

### Visualization (`src/visualization.py`)

Handles all visualization components:
- Generates interactive graph visualizations
- Creates statistical charts and plots
- Displays entity details
- Visualizes semantic paths and reasoning traces

## Deployment

For detailed deployment instructions, please refer to the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Performance Considerations

- **Memory Usage**: The application loads the entire ontology into memory, which can be significant for large ontologies
- **Computation Intensity**: Graph operations and path finding can be computationally expensive
- **API Rate Limits**: The application makes calls to the OpenAI API, which has rate limits and costs
- **Visualization Rendering**: Complex graphs may affect browser performance

## Acknowledgments

This project demonstrates the integration of ontological knowledge with RAG systems for enhanced query answering capabilities. It builds upon research in knowledge graphs, semantic web technologies, and large language models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
