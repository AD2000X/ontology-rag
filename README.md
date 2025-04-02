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

# Enhanced Ontology-RAG System

## Project Overview

This repository contains an advanced Retrieval-Augmented Generation (RAG) system that integrates structured ontologies with language models. The system demonstrates how formal ontological knowledge representation can enhance traditional vector-based retrieval methods to provide more accurate, contextually rich, and logically consistent answers to user queries.

The project implements a sophisticated architecture that combines:

- JSON-based ontology representation with classes, relationships, rules, and instances  
- Knowledge graph visualization for exploring entity relationships  
- Semantic path finding for multi-hop reasoning between concepts  
- Comparative analysis between traditional vector-based RAG and ontology-enhanced RAG  

The application is built with **Streamlit** for the frontend interface, uses **FAISS** for vector embeddings, **NetworkX** for graph representation, and integrates with **OpenAI's language models** for generating responses.

## Key Features

1. **RAG Comparison Demo**
   - Side-by-side comparison of traditional and ontology-enhanced RAG
   - Analysis of differences in answers and retrieved context

2. **Knowledge Graph Visualization**
   - Interactive network graph for exploring the ontology structure
   - Multiple layout algorithms (force-directed, hierarchical, radial, circular)
   - Entity relationship exploration with customizable focus

3. **Ontology Structure Analysis**
   - Visualization of class hierarchies and statistics
   - Relationship usage and domain-range distribution analysis
   - Graph statistics including node counts, edge counts, and centrality metrics

4. **Entity Exploration**
   - Detailed entity information cards showing properties and relationships
   - Relationship graphs centered on specific entities
   - Neighborhood exploration for entities

5. **Semantic Path Visualization**
   - Path visualization between entities with step-by-step explanation
   - Visual representation of paths through the knowledge graph
   - Connection to relevant business rules

6. **Reasoning Trace Visualization**
   - Query analysis with entity and relationship detection
   - Sankey diagrams showing information flow in the RAG process
   - Explanation of reasoning steps

## Ontology Structure Example

The `data/enterprise_ontology.json` file contains a rich enterprise ontology that models organizational knowledge. Here's a breakdown of its key components:

### Classes (Entity Types)

The ontology defines a hierarchical class structure with inheritance relationships. For example:

- **Entity** (base class)
  - **FinancialEntity** → Budget, Revenue, Expense  
  - **Asset** → PhysicalAsset, DigitalAsset, IntellectualProperty  
  - **Person** → InternalPerson → Employee → Manager  
  - **Process** → BusinessProcess, DevelopmentProcess, SupportProcess  
  - **Market** → GeographicMarket, DemographicMarket, BusinessMarket  

Each class has a description and a set of defined properties. For instance, the `Employee` class includes properties like role, hire date, and performance rating.

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

Each instance has properties and relationships to other instances, forming a connected knowledge graph.

This structured knowledge representation allows the system to perform semantic reasoning beyond what would be possible with simple text-based approaches, enabling it to answer complex queries that require understanding of hierarchical relationships, business rules, and multi-step connections between entities.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key as an environment variable or in the Streamlit secrets

### Running the Application

To run the application locally:

```
streamlit run app.py
```

For deployment instructions, please refer to the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Project Structure

```
ontology-rag/
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── data/
│   └── enterprise_ontology.json  # Enterprise ontology data
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

## Use Cases

### Enterprise Knowledge Management
The ontology-enhanced RAG system can help organizations effectively organize and access their knowledge assets, connecting information across different departments and systems to provide more comprehensive business insights.

### Product Development Decision Support
By understanding the relationships between customer feedback, product features, and market data, the system can provide more valuable support for product development decisions.

### Complex Compliance Queries
In compliance scenarios where multiple rules and relationships need to be considered, the ontology-enhanced RAG can provide rule-based reasoning to ensure recommendations comply with all applicable policies and regulations.

### Diagnostics and Troubleshooting
In technical support and troubleshooting scenarios, the system can connect symptoms, causes, and solutions through multi-hop reasoning to provide more accurate diagnoses.

## Acknowledgments

This project demonstrates the integration of ontological knowledge with RAG systems for enhanced query answering capabilities. It builds upon research in knowledge graphs, semantic web technologies, and large language models.

## License

This project is licensed under the MIT License - see the license file for details.