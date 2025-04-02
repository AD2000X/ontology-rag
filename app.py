import streamlit as st
st.set_page_config(page_title="Ontology RAG Demo", layout="wide")

import os
from src.semantic_retriever import SemanticRetriever
from src.ontology_manager import OntologyManager
from src.knowledge_graph import KnowledgeGraph
from src.visualization import (display_ontology_stats, display_entity_details, 
                              display_graph_visualization, visualize_path, 
                              display_reasoning_trace, render_html_in_streamlit)
import networkx as nx
from openai import OpenAI
import json

# Setup
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
ontology_manager = OntologyManager("data/enterprise_ontology.json")
semantic_retriever = SemanticRetriever(ontology_manager=ontology_manager)
knowledge_graph = KnowledgeGraph(ontology_manager=ontology_manager)
k_val = st.sidebar.slider("Top K Results", 1, 10, 3)

def main():
    # Page Navigation
    st.sidebar.title("Page Navigation")
    page = st.sidebar.selectbox(
        "Select function", 
        ["RAG comparison demonstration", "Knowledge graph visualization", "Ontology structure analysis", "Entity exploration", "Semantic path visualization", "Inference tracking", "Detailed comparative analysis"]
    )
    
    # # Fix the conditional judgment to make it consistent with the option name
    if page == "RAG comparison demonstration":
        run_rag_demo()
    elif page == "Knowledge graph visualization":
        run_knowledge_graph_visualization()
    elif page == "Ontology structure analysis":
        run_ontology_structure_analysis()
    elif page == "Entity exploration":
        run_entity_exploration()
    elif page == "Semantic path visualization":
        run_semantic_path_visualization()
    elif page == "Inference tracking":
        run_reasoning_trace()
    elif page == "Detailed comparative analysis":
        run_detailed_comparison()
    
def run_rag_demo():
    st.title("Ontology Enhanced RAG Demonstration")
    
    query = st.text_input(
        "Enter a question to compare RAG methods:",
        "How does customer feedback influence product development?"
    )

    if query:
        col1, col2 = st.columns(2)
        
        with st.spinner("Run two RAG methods..."):
            # Traditional RAG
            with col1:
                st.subheader("Traditional RAG")
                vector_docs = semantic_retriever.vector_store.similarity_search(query, k=k_val)
                vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
                vector_messages = [
                    {"role": "system", "content": f"You are an enterprise knowledge assistant...\nContext:\n{vector_context}"},
                    {"role": "user", "content": query}
                ]
                vector_response = llm.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=vector_messages
                )
                vector_answer = vector_response.choices[0].message.content

                st.markdown("#### Answer")
                st.write(vector_answer)

                st.markdown("#### Retrieved Context")
                for i, doc in enumerate(vector_docs):
                    with st.expander(f"Source {i+1}"):
                        st.code(doc.page_content)

            # Ontology RAG
            with col2:
                st.subheader("Ontology RAG")
                result = semantic_retriever.retrieve_with_paths(query, k=k_val)
                retrieved_docs = result["documents"]
                enhanced_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                enhanced_messages = [
                    {"role": "system", "content": f"You are an enterprise knowledge assistant with ontology access rights...\nContext:\n{enhanced_context}"},
                    {"role": "user", "content": query}
                ]
                enhanced_response = llm.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=enhanced_messages
                )
                enhanced_answer = enhanced_response.choices[0].message.content

                st.markdown("#### Answer")
                st.write(enhanced_answer)

                st.markdown("#### Retrieved Context")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source", "unknown")
                    label = {
                        "ontology": "Ontology Context",
                        "text": "Text Context",
                        "ontology_context": "Semantic Context",
                        "semantic_path": "Relationship Path"
                    }.get(source, f"Source")
                    with st.expander(f"{label} {i+1}"):
                        st.markdown(doc.page_content)
                
                # Store for reasoning trace visualization
                st.session_state.query = query
                st.session_state.retrieved_docs = retrieved_docs
                st.session_state.answer = enhanced_answer
        
        # Difference Analysis
        st.markdown("---")
        st.subheader("Difference Analysis")
        
        st.markdown("""
        The above comparison demonstrates several key advantages of ontology-enhanced RAG:
        
        1. **Structural Awareness**: The ontology-enhanced approach understands the relationships between entities, not just their textual similarity.
        
        2. **Multi-hop Reasoning**: By using the knowledge graph structure, the enhanced approach can connect information across multiple relationship hops.
        
        3. **Context Enrichment**: The ontology provides additional context about entity types, properties, and relationships that isn't explicit in the text.
        
        4. **Inference Capabilities**: The structured knowledge allows for logical inferences that vector similarity alone cannot achieve.
        
        Try more complex queries that require understanding relationships to see the differences more clearly!
        """)

def run_knowledge_graph_visualization():
    st.title("Knowledge Graph Visualization")
    
    # Check if there is a central entity selected
    central_entity = st.session_state.get('central_entity', None)
    
    # Call visualization function
    display_graph_visualization(knowledge_graph, central_entity=central_entity, max_distance=2)
    
    # Get and display graph statistics
    graph_stats = knowledge_graph.get_graph_statistics()
    if graph_stats:
        st.subheader("Graph Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", graph_stats.get("node_count", 0))
        col2.metric("Edges", graph_stats.get("edge_count", 0))
        col3.metric("Classes", graph_stats.get("class_count", 0))
        col4.metric("Instances", graph_stats.get("instance_count", 0))
        
        # Display central nodes
        if "central_nodes" in graph_stats and graph_stats["central_nodes"]:
            st.subheader("Central Nodes (by Betweenness Centrality)")
            central_nodes = graph_stats["central_nodes"]["betweenness"]
            nodes_df = []
            for node_info in central_nodes:
                node_id = node_info["node"]
                node_data = knowledge_graph.graph.nodes.get(node_id, {})
                node_type = node_data.get("type", "unknown")
                if node_type == "instance":
                    node_class = node_data.get("class_type", "unknown")
                    properties = node_data.get("properties", {})
                    name = properties.get("name", node_id)
                    nodes_df.append({
                        "ID": node_id,
                        "Name": name,
                        "Type": node_class,
                        "Centrality": node_info["centrality"]
                    })
            
            st.table(nodes_df)

def run_ontology_structure_analysis():
    st.title("Ontology Structure Analysis")
    
    # Use ontology statistics display function
    display_ontology_stats(ontology_manager)
    
    # Add class hierarchy visualization
    st.subheader("Class Hierarchy")
    
    # Get class hierarchy data
    class_hierarchy = ontology_manager.get_class_hierarchy()
    
    # Create a NetworkX graph to represent the class hierarchy
    G = nx.DiGraph()
    
    # Add nodes and edges
    for parent, children in class_hierarchy.items():
        if not G.has_node(parent):
            G.add_node(parent)
        for child in children:
            G.add_node(child)
            G.add_edge(parent, child)
    
    # Check if there are enough nodes to create visualization
    if len(G.nodes) > 1:
        # Generate HTML visualization using knowledge graph class
        kg = KnowledgeGraph(ontology_manager)
        
        # Use built-in layout algorithm
        html = kg.generate_html_visualization(
            include_classes=True,
            include_instances=False,
            max_distance=5,
            layout_algorithm="hierarchical"  # Use the built-in hierarchical layout
        )
        
        # Render HTML
        render_html_in_streamlit(html)
        
        # Add extra tree view for each root node
        with st.expander("Node Tree View", expanded=False):
            # Find root nodes (nodes without parent nodes)
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            # Display tree structure for each root node
            for root in roots:
                st.markdown(f"### Root Node: {root}")
                
                # Recursively display child nodes
                def display_tree(node, depth=0):
                    children = list(G.successors(node))
                    if children:
                        for child in sorted(children):
                            st.markdown("&nbsp;" * depth * 4 + f"- {child}")
                            display_tree(child, depth + 1)
                
                display_tree(root)
                st.markdown("---")

def run_entity_exploration():
    st.title("Entity Exploration")

    # Grab all entities from the graph
    entities = [
        node for node, attr in ontology_manager.graph.nodes(data=True)
        if attr.get("type") == "instance"
    ]
    entities = sorted(set(entities))

    selected_entity = st.selectbox("Select Entity", entities) if entities else None

    if selected_entity:
        entity_info = ontology_manager.get_entity_info(selected_entity)
        display_entity_details(entity_info, ontology_manager)

        if st.button("View this Entity in the Knowledge Graph", key=f"view_entity_{selected_entity}"):
            st.session_state.central_entity = selected_entity
            st.rerun()

        st.subheader("Entity Neighborhood")
        max_distance = st.slider("Maximum Neighborhood Distance", 1, 3, 1, key=f"distance_slider_{selected_entity}")
        neighborhood = knowledge_graph.get_entity_neighborhood(
            selected_entity,
            max_distance=max_distance,
            include_classes=True
        )

        if neighborhood and "neighbors" in neighborhood:
            lines = []
            for distance in range(1, max_distance+1):
                neighbors_at_distance = [n for n in neighborhood["neighbors"] if n["distance"] == distance]

                if neighbors_at_distance:
                    lines.append(f"**Neighbors at Distance {distance} ({len(neighbors_at_distance)})**")
                    for neighbor in neighbors_at_distance:
                        lines.append(f"- **{neighbor['id']}** ({neighbor.get('class_type', 'unknown')})")
                        for relation in neighbor.get("relations", []):
                            direction = "→" if relation["direction"] == "outgoing" else "←"
                            lines.append(f"  - {direction} {relation['type']}")
                        lines.append("---")
            with st.expander("Show Neighbor Details", expanded=False):
                for line in lines:
                    st.markdown(line)

    elif not entities:
        st.warning("No entities found in the ontology.")

def render_semantic_path_tab():
    st.title("Semantic Path Visualization")

    entities = [
        node for node, attr in ontology_manager.graph.nodes(data=True)
        if attr.get("type") == "instance"
    ]
    entities = sorted(set(entities))

    source_entity = st.selectbox("Select Source Entity", entities, key="source_entity") if entities else None
    target_entity = st.selectbox("Select Target Entity", entities, key="target_entity") if entities else None

    if source_entity and target_entity and source_entity != target_entity:
        max_length = st.slider("Maximum Path Length", 1, 5, 3)

        paths = knowledge_graph.find_paths_between_entities(
            source_entity,
            target_entity,
            max_length=max_length
        )

        if paths:
            st.success(f"Found {len(paths)} paths!")
            for i, path in enumerate(paths):
                path_length = len(path)
                rel_types = [edge["type"] for edge in path]

                with st.expander(f"Path {i+1} (Length: {path_length}, Relations: {', '.join(rel_types)})", expanded=(i==0)):
                    path_text = []
                    for edge in path:
                        source = edge["source"]
                        target = edge["target"]
                        relation = edge["type"]

                        source_info = ontology_manager.get_entity_info(source)
                        target_info = ontology_manager.get_entity_info(target)

                        source_name = source_info.get("properties", {}).get("name", source)
                        target_name = target_info.get("properties", {}).get("name", target)

                        path_text.append(f"{source_name} ({source}) **{relation}** {target_name} ({target})")

                    st.markdown(" → ".join(path_text))

                    path_info = {
                        "source": source_entity,
                        "target": target_entity,
                        "path": path,
                        "text": " → ".join(path_text)
                    }

                    # Render full visualization outside nested expander
                    st.subheader("Path Visualization")
                    visualize_path(path_info, ontology_manager)
        else:
            st.warning(f"No paths of length {max_length} or shorter were found between these entities.")
    elif not entities:
        st.warning("No entities available for semantic path selection.")

# Alias for compatibility with app.py call site
run_semantic_path_visualization = render_semantic_path_tab


def run_reasoning_trace():
    st.title("Reasoning Trace Visualization")
    
    if not st.session_state.get("query") or not st.session_state.get("retrieved_docs") or not st.session_state.get("answer"):
        st.warning("Please run a query on the RAG comparison page first to generate reasoning trace data.")
        return
    
    # Get data from session state
    query = st.session_state.query
    retrieved_docs = st.session_state.retrieved_docs
    answer = st.session_state.answer
    
    # Show reasoning trace
    display_reasoning_trace(query, retrieved_docs, answer, ontology_manager)

def run_detailed_comparison():
    st.title("Detailed Comparison of RAG Methods")
    
    # Add comparison query options
    comparison_queries = [
        "How does customer feedback influence product development?",
        "Which employees work in the Engineering department?",
        "What are the product life cycle stages?",
        "How do managers monitor employee performance?",
        "What are the responsibilities of the marketing department?"
    ]
    
    selected_query = st.selectbox(
        "Select Comparison Query", 
        comparison_queries,
        index=0
    )
    
    custom_query = st.text_input("Or enter a custom query:", "")
    
    if custom_query:
        query = custom_query
    else:
        query = selected_query
    
    if st.button("Compare RAG Methods"):
        with st.spinner("Running detailed comparison..."):
            # Start timing
            import time
            start_time = time.time()
            
            # Run traditional RAG
            vector_docs = semantic_retriever.vector_store.similarity_search(query, k=k_val)
            vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
            vector_messages = [
                {"role": "system", "content": f"You are an enterprise knowledge assistant...\nContext:\n{vector_context}"},
                {"role": "user", "content": query}
            ]
            vector_response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=vector_messages
            )
            vector_answer = vector_response.choices[0].message.content
            vector_time = time.time() - start_time
            
            # Reset timer
            start_time = time.time()
            
            # Run ontology-enhanced RAG
            result = semantic_retriever.retrieve_with_paths(query, k=k_val)
            retrieved_docs = result["documents"]
            enhanced_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            enhanced_messages = [
                {"role": "system", "content": f"You are an enterprise knowledge assistant with ontology access rights...\nContext:\n{enhanced_context}"},
                {"role": "user", "content": query}
            ]
            enhanced_response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=enhanced_messages
            )
            enhanced_answer = enhanced_response.choices[0].message.content
            enhanced_time = time.time() - start_time
            
            # Save results for visualization
            st.session_state.query = query
            st.session_state.retrieved_docs = retrieved_docs
            st.session_state.answer = enhanced_answer
            
            # Display comparison results
            st.subheader("Comparison Results")
            
            # Use tabs to show different aspects of comparison
            tab1, tab2, tab3, tab4 = st.tabs(["Answer Comparison", "Performance Metrics", "Retrieval Source Comparison", "Context Quality"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Traditional RAG Answer")
                    st.write(vector_answer)
                
                with col2:
                    st.markdown("#### Ontology-Enhanced RAG Answer")
                    st.write(enhanced_answer)
            
            with tab2:
                # Performance metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Traditional RAG Response Time", f"{vector_time:.2f} seconds")
                    
                    # Calculate text metrics
                    vector_tokens = len(vector_context.split())
                    st.metric("Retrieved Context Tokens", vector_tokens)
                    
                    st.metric("Retrieved Documents", len(vector_docs))
                
                with col2:
                    st.metric("Ontology-Enhanced RAG Response Time", f"{enhanced_time:.2f} seconds")
                    
                    # Calculate text metrics
                    enhanced_tokens = len(enhanced_context.split())
                    st.metric("Retrieved Context Tokens", enhanced_tokens)
                    
                    st.metric("Retrieved Documents", len(retrieved_docs))
                
                # Add chart
                import pandas as pd
                import plotly.express as px
                
                # Performance comparison chart
                performance_data = {
                    "Metrics": ["Response Time (seconds)", "Context Tokens", "Retrieved Documents"],
                    "Traditional RAG": [vector_time, vector_tokens, len(vector_docs)],
                    "Ontology-Enhanced RAG": [enhanced_time, enhanced_tokens, len(retrieved_docs)]
                }
                
                df = pd.DataFrame(performance_data)
                
                # Plotly bar chart
                fig = px.bar(
                    df, 
                    x="Metrics",  # Fixed column name
                    y=["Traditional RAG", "Ontology-Enhanced RAG"],
                    barmode="group",
                    title="Performance Metrics Comparison",
                    labels={"value": "Value", "variable": "RAG Method"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Retrieval source comparison
                traditional_sources = ["Traditional Vector Retrieval"] * len(vector_docs)
                
                enhanced_sources = []
                for doc in retrieved_docs:
                    source = doc.metadata.get("source", "unknown")
                    label = {
                        "ontology": "Ontology Context",
                        "text": "Text Context",
                        "ontology_context": "Semantic Context",
                        "semantic_path": "Relationship Path"
                    }.get(source, "Unknown Source")
                    enhanced_sources.append(label)
                
                # Create source distribution chart
                source_counts = {}
                for source in enhanced_sources:
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
                
                source_df = pd.DataFrame({
                    "Source Type": list(source_counts.keys()),
                    "Document Count": list(source_counts.values())
                })
                
                fig = px.pie(
                    source_df,
                    values="Document Count",
                    names="Source Type",
                    title="Ontology-Enhanced RAG Retrieval Source Distribution"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show source-answer relationship
                st.subheader("Relationship Between Sources and Answer")
                st.markdown("""
                Ontology-enhanced methods leverage multiple sources of knowledge to construct more comprehensive answers. The figure above shows the distribution of different sources.
                
                In particular, semantic context and relationship paths provide knowledge that cannot be captured by traditional vector retrieval, enabling the system to connect concepts and perform multi-hop reasoning.
                """)
            
            with tab4:
                # Context quality assessment
                st.subheader("Context Quality Assessment")
                
                # Create evaluation function (simplified)
                def evaluate_context(docs):
                    metrics = {
                        "Direct Relevance": 0,
                        "Semantic Richness": 0,
                        "Structure Information": 0,
                        "Relationship Information": 0
                    }
                    
                    for doc in docs:
                        content = doc.page_content if hasattr(doc, "page_content") else ""
                        
                        # Direct Relevance - Based on Keywords
                        if any(kw in content.lower() for kw in query.lower().split()):
                            metrics["Direct Relevance"] += 1
                        
                        # Semantic richness - based on text length
                        metrics["Semantic Richness"] += min(1, len(content.split()) / 50)
                        
                        # Structural information - from ontology
                        if hasattr(doc, "metadata") and doc.metadata.get("source") in ["ontology", "ontology_context"]:
                            metrics["Structure Information"] += 1
                        
                        # Relationship information - from path
                        if hasattr(doc, "metadata") and doc.metadata.get("source") == "semantic_path":
                            metrics["Relationship Information"] += 1
                    
                    # Standardization
                    for key in metrics:
                        metrics[key] = min(10, metrics[key])
                    
                    return metrics
                
                # Evaluate both methods
                vector_metrics = evaluate_context(vector_docs)
                enhanced_metrics = evaluate_context(retrieved_docs)
                
                # Create comparative radar chart
                metrics_df = pd.DataFrame({
                    "metrics": list(vector_metrics.keys()),
                    "Traditional RAG": list(vector_metrics.values()),
                    "Ontology-Enhanced RAG": list(enhanced_metrics.values())
                })
                
                # # Convert wide-format to long-format
                plot_df = metrics_df.melt(
                    id_vars=["metrics"],
                    value_vars=["Traditional RAG", "Ontology-Enhanced RAG"],
                    var_name="Method",
                    value_name="Value"
                )
                
                # Type confirmation (optional but stable)
                plot_df = plot_df.astype({"metrics": str, "Value": float, "Method": str})
                
                # draw bar_polar graph
                fig = px.bar_polar(
                    plot_df,
                    r="Value",
                    theta="metrics",
                    color="Method",
                    title="Context Quality Assessment (Polar View)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                           
                st.markdown("""
                                The figure above shows a comparison of the two RAG methods in terms of context quality. Ontology-enhanced RAG performs better in multiple dimensions:
                                
                                1. **Direct Relevance**: The degree of relevance between the retrieved content and the query
                                2. **Semantic Richness**: Information density and richness of the retrieval context
                                3. **Structural Information**: Structured knowledge of entity types, attributes, and relationships
                                4. **Relationship Information**: Explicit relationships and connection paths between entities
                                
                                The advantage of ontology-enhanced RAG is that it can retrieve structured knowledge and relational information, which are missing in traditional RAG methods.
                                """)
                        
            # Display detailed analysis section
            st.subheader("Method Effectiveness Analysis")
            
            with st.expander("Comparison of Advantages and Disadvantages", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Traditional RAG")
                    st.markdown("""
                    **Advantages**:
                    - Simple implementation and light computational burden
                    - Works well with unstructured text
                    - Response times are usually faster
                    
                    **Disadvantages**:
                    - Unable to capture relationships between entities
                    - Lack of context for structured knowledge
                    - Difficult to perform multi-hop reasoning
                    - Retrieval is mainly based on text similarity
                    """)
                
                with col2:
                    st.markdown("#### Ontology Enhanced RAG")
                    st.markdown("""
                    **Advantages**:
                    - Ability to understand relationships and connections between entities
                    - Provides rich structured knowledge context
                    - Support multi-hop reasoning and path discovery
                    - Combining vector similarity and semantic relationship
                    
                    **Disadvantages**:
                    - Higher implementation complexity
                    - Need to maintain the ontology model
                    - The computational overhead is relatively high
                    - Retrieval and inference times may be longer
                    """)
            
            # Add usage scenario suggestions
            with st.expander("Applicable Scenarios"):
                st.markdown("""
                ### Traditional RAG Applicable Scenarios
                
                - Simple fact-finding
                - Unstructured document retrieval
                - Applications with high response time requirements
                - When the document content is clear and direct
                
                ### Applicable Scenarios for Ontology Enhanced RAG
                
                - Complex knowledge association query
                - Problems that require understanding of relationships between entities
                - Applications that require cross-domain reasoning
                - Enterprise Knowledge Management System
                - Reasoning scenarios that require high accuracy and consistency
                - Applications that require implicit knowledge discovery
                """)

            # Add practical application examples
            with st.expander("Application Case Studies"):
                st.markdown("""
                ### Enterprise Knowledge Management
                Ontology-enhanced RAG systems can help enterprises effectively organize and access their knowledge assets, connect information in different departments and systems, and provide more comprehensive business insights.
                
                ### Product Development Decision Support
                By understanding the relationship between customer feedback, product features, and market data, the system can provide more valuable support for product development decisions.
                
                ### Complex Compliance Queries
                In compliance problems that require consideration of multiple rules and relationships, ontology-enhanced RAG can provide rule-based reasoning, ensuring that recommendations comply with all applicable policies and regulations.
                
                ### Diagnostics and Troubleshooting
                In technical support and troubleshooting scenarios, the system can connect symptoms, causes, and solutions to provide more accurate diagnoses through multi-hop reasoning.
                """)

if __name__ == "__main__":
    main()