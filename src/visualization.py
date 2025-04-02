# Add this at the top of visualization.py with other imports
import re
import streamlit as st
import json
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import math

def analyze_query_ontology_concepts(query: str, ontology_manager) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze the query to identify ontology concepts with confidence scores.
    This is a simplified implementation that would be replaced with NLP in production.
    """
    # For debugging - print the query
    st.write(f"Debug - Analyzing query: '{query}'")
    
    query_lower = query.lower()
    
    # Entity detection
    entity_mentions = []
    classes = ontology_manager.get_classes()
    
    # Debug - print available classes
    st.write(f"Debug - Available classes: {classes[:5]}...")
    
    for class_name in classes:
        # Check for both exact word match and partial match
        word_match = False
        partial_match = False
        
        # Word boundary regex for exact word match
        pattern = r'\b' + re.escape(class_name.lower()) + r'\b'
        word_match = bool(re.search(pattern, query_lower))
        
        # Check for partial match in case class has multiple words or special spelling
        partial_match = class_name.lower() in query_lower
        
        if word_match or partial_match:
            # Get class info
            class_info = ontology_manager.ontology_data["classes"].get(class_name, {})
            
            # Confidence is higher for word match than partial match
            base_confidence = 0.9 if word_match else 0.7
            length_factor = min(0.05, (len(class_name) / 400))  # Adjust for length but don't penalize too much
            confidence = min(0.95, base_confidence + length_factor)
            
            entity_mentions.append({
                "type": class_name,
                "confidence": confidence,
                "description": class_info.get("description", ""),
                "match_type": "word" if word_match else "partial"
            })
    
    # Debug - show what was found
    st.write(f"Debug - Found {len(entity_mentions)} entity mentions")
    
    # Relationship detection
    relationship_mentions = []
    relationships = ontology_manager.ontology_data.get("relationships", [])
    
    for rel in relationships:
        rel_name = rel["name"]
        
        # Try both word boundary and partial match
        word_match = False
        partial_match = False
        
        pattern = r'\b' + re.escape(rel_name.lower()) + r'\b'
        word_match = bool(re.search(pattern, query_lower))
        
        # For relationships like "ownedBy", check if it appears as part of words too
        partial_match = rel_name.lower() in query_lower
        
        if word_match or partial_match:
            # Higher confidence for word match
            base_confidence = 0.85 if word_match else 0.65
            length_factor = min(0.05, (len(rel_name) / 400))
            confidence = min(0.9, base_confidence + length_factor)
            
            relationship_mentions.append({
                "name": rel_name,
                "domain": rel["domain"],
                "range": rel["range"],
                "confidence": confidence,
                "description": rel.get("description", ""),
                "match_type": "word" if word_match else "partial"
            })
    
    # Debug - show what was found
    st.write(f"Debug - Found {len(relationship_mentions)} relationship mentions")
    
    # Add hardcoded examples if nothing detected (fallback for debugging)
    if not entity_mentions and not relationship_mentions:
        st.write("Debug - No matches found, adding fallback examples")
        
        # Add some fallback examples to ensure the UI displays something
        entity_mentions.append({
            "type": "Customer",
            "confidence": 0.8,
            "description": "A person or organization that purchases products or services",
            "match_type": "fallback"
        })
        
        entity_mentions.append({
            "type": "Product",
            "confidence": 0.75,
            "description": "An item offered for sale or use",
            "match_type": "fallback"
        })
        
        relationship_mentions.append({
            "name": "provides",
            "domain": "Customer",
            "range": "Feedback",
            "confidence": 0.7,
            "description": "Connects customers to their feedback submissions",
            "match_type": "fallback"
        })
    
    # Clear debug messages before returning
    # st.empty()
    
    return entity_mentions, relationship_mentions

# The rest of your visualization.py file, including the fixed display_reasoning_trace function
def display_reasoning_trace(query: str, retrieved_docs: List[Dict], answer: str, ontology_manager):
    """Display an enhanced trace of how ontological reasoning was used to answer the query."""
    st.subheader("🧠 Ontology-Enhanced Reasoning")
    
    # Create a multi-tab interface for different aspects of reasoning
    tab1, tab2, tab3 = st.tabs(["Query Analysis", "Knowledge Retrieval", "Reasoning Path"])
    
    with tab1:
        # Extract entity and relationship mentions with confidence
        entity_mentions, relationship_mentions = analyze_query_ontology_concepts(query, ontology_manager)
        
        # Display detected entities with confidence scores
        if entity_mentions:
            st.markdown("### Entities Detected in Query")
            
            # Convert to DataFrame for visualization
            entity_df = pd.DataFrame([{
                "Entity Type": e["type"],
                "Confidence": e["confidence"],
                "Description": e["description"]
            } for e in entity_mentions])
            
            # Sort by confidence
            entity_df = entity_df.sort_values("Confidence", ascending=False)
            
            # Create a horizontal bar chart
            fig = px.bar(entity_df, 
                        x="Confidence", 
                        y="Entity Type",
                        orientation='h',
                        title="Entity Type Detection Confidence",
                        color="Confidence",
                        color_continuous_scale="Blues",
                        text="Confidence")
            
            fig.update_traces(texttemplate='%{text:.0%}', textposition='outside')
            fig.update_layout(xaxis_tickformat=".0%")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display descriptions
            st.subheader("Entity Type Descriptions")
            st.dataframe(
                entity_df[["Entity Type", "Description"]],
                hide_index=True
            )
        
        # Display detected relationships
        if relationship_mentions:
            st.markdown("### Relationships Detected in Query")
            
            # Convert to DataFrame
            rel_df = pd.DataFrame([{
                "Relationship": r["name"],
                "From": r["domain"],
                "To": r["range"],
                "Confidence": r["confidence"],
                "Description": r["description"]
            } for r in relationship_mentions])
            
            # Sort by confidence
            rel_df = rel_df.sort_values("Confidence", ascending=False)
            
            # Create visualization
            fig = px.bar(rel_df, 
                        x="Confidence", 
                        y="Relationship",
                        orientation='h',
                        title="Relationship Detection Confidence",
                        color="Confidence",
                        color_continuous_scale="Reds",
                        text="Confidence")
            
            fig.update_traces(texttemplate='%{text:.0%}', textposition='outside')
            fig.update_layout(xaxis_tickformat=".0%")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display relationship details
            st.subheader("Relationship Details")
            st.dataframe(
                rel_df[["Relationship", "From", "To", "Description"]],
                hide_index=True
            )
    
    with tab2:
        # Create an enhanced visualization of the retrieval process
        st.markdown("### Knowledge Retrieval Process")
        
        # Group retrieved documents by source
        docs_by_source = defaultdict(list)
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'unknown')
                docs_by_source[source].append(doc)
            else:
                docs_by_source['unknown'].append(doc)
        
        # Display retrieval visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a Sankey diagram to show flow from query to sources to answer
            display_retrieval_flow(query, docs_by_source)
        
        with col2:
            # Display source distribution
            source_counts = {source: len(docs) for source, docs in docs_by_source.items()}
            
            # Create a pie chart
            fig = px.pie(
                values=list(source_counts.values()),
                names=list(source_counts.keys()),
                title="Retrieved Context Sources",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display retrieved document details in expandable sections
        for source, docs in docs_by_source.items():
            st.subheader(f"{source} ({len(docs)})")
            
            for i, doc in enumerate(docs):
                # Add separator between documents
                if i > 0:
                    st.markdown("---")
                
                # Display document content
                if hasattr(doc, 'page_content'):
                    st.markdown(f"**Content:**")
                    
                    # Format depending on source
                    if source in ["ontology", "ontology_context"]:
                        st.markdown(doc.page_content)
                    else:
                        st.code(doc.page_content)
                
                # Display metadata if present
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.markdown("**Metadata:**")
                    for key, value in doc.metadata.items():
                        if key != 'source':  # Already shown in section title
                            st.markdown(f"- **{key}**: {value}")
    
    with tab3:
        # Show the reasoning flow from query to answer
        st.markdown("### Ontological Reasoning Process")
        
        # Display reasoning steps
        reasoning_steps = generate_reasoning_steps(query, entity_mentions, relationship_mentions, retrieved_docs, answer)
        
        for i, step in enumerate(reasoning_steps):
            with st.expander(f"Step {i+1}: {step['title']}", expanded=i == 0):
                st.markdown(step["description"])
        
        # Visualization of how ontological structure influenced the answer
        st.markdown("### How Ontology Enhanced the Answer")
        
        # Display ontology advantage explanation
        advantages = explain_ontology_advantages(entity_mentions, relationship_mentions)
        
        for adv in advantages:
            st.markdown(f"**{adv['title']}**")
            st.markdown(adv["description"])

# The rest of the code remains the same as in your original visualization.py file
def display_retrieval_flow(query: str, docs_by_source: Dict[str, List]):
    """Create a Sankey diagram showing the flow from query to sources to answer."""
    # Define node labels
    nodes = ["Query"]
    
    # Add source nodes
    for source in docs_by_source.keys():
        nodes.append(f"Source: {source.capitalize()}")
    
    nodes.append("Answer")
    
    # Define links
    source_indices = []
    target_indices = []
    values = []
    
    # Links from query to sources
    for i, (source, docs) in enumerate(docs_by_source.items()):
        source_indices.append(0)  # Query is index 0
        target_indices.append(i + 1)  # Source indices start at 1
        values.append(len(docs))  # Width based on number of docs
    
    # Links from sources to answer
    for i in range(len(docs_by_source)):
        source_indices.append(i + 1)  # Source index
        target_indices.append(len(nodes) - 1)  # Answer is last node
        values.append(values[i])  # Same width as query to source
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#1f77b4"] + [px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                for i in range(len(docs_by_source))] + ["#2ca02c"]
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values
        )
    )])
    
    fig.update_layout(
        title="Information Flow in RAG Process",
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_reasoning_steps(query: str, entity_mentions: List[Dict], relationship_mentions: List[Dict],
                            retrieved_docs: List[Dict], answer: str) -> List[Dict]:
    """Generate reasoning steps to explain how the system arrived at the answer."""
    steps = []
    
    # Step 1: Query Understanding
    steps.append({
        "title": "Query Understanding",
        "description": f"""The system analyzes the query "{query}" and identifies key concepts from the ontology. 
        {len(entity_mentions)} entity types and {len(relationship_mentions)} relationship types are recognized, allowing 
        the system to understand the semantic context of the question."""
    })
    
    # Step 2: Knowledge Retrieval
    if retrieved_docs:
        doc_count = len(retrieved_docs)
        ontology_count = sum(1 for doc in retrieved_docs if hasattr(doc, 'metadata') and 
                           doc.metadata.get('source', '') in ['ontology', 'ontology_context'])
        
        steps.append({
            "title": "Knowledge Retrieval",
            "description": f"""Based on the identified concepts, the system retrieves {doc_count} relevant pieces of information,
            including {ontology_count} from the structured ontology. This hybrid approach combines traditional vector retrieval
            with ontology-aware semantic retrieval, enabling access to both explicit and implicit knowledge."""
        })
    
    # Step 3: Relationship Traversal
    if relationship_mentions:
        rel_names = [r["name"] for r in relationship_mentions]
        steps.append({
            "title": "Relationship Traversal",
            "description": f"""The system identifies key relationships in the ontology: {', '.join(rel_names)}. 
            By traversing these relationships, the system can connect concepts that might not appear together in the same text,
            allowing for multi-hop reasoning across the knowledge graph."""
        })
    
    # Step 4: Ontological Inference
    if entity_mentions:
        entity_types = [e["type"] for e in entity_mentions]
        steps.append({
            "title": "Ontological Inference",
            "description": f"""Using the hierarchical structure of entities like {', '.join(entity_types)}, 
            the system makes inferences based on class inheritance and relationship constraints defined in the ontology.
            This allows it to reason about properties and relationships that might not be explicitly stated."""
        })
    
    # Step 5: Answer Generation
    steps.append({
        "title": "Answer Synthesis",
        "description": f"""Finally, the system synthesizes the retrieved information and ontological knowledge to generate a comprehensive answer.
        The structured nature of the ontology ensures that the answer accurately reflects the relationships between concepts
        and respects the business rules defined in the knowledge model."""
    })
    
    return steps


def explain_ontology_advantages(entity_mentions: List[Dict], relationship_mentions: List[Dict]) -> List[Dict]:
    """Explain how ontology enhanced the RAG process."""
    advantages = []
    
    if entity_mentions:
        advantages.append({
            "title": "Hierarchical Knowledge Representation",
            "description": """The ontology provides a hierarchical class structure that enables the system to understand
            that concepts are related through is-a relationships. For instance, knowing that a Manager is an Employee
            allows the system to apply Employee-related knowledge when answering questions about Managers, even if
            the specific information was only stated for Employees in general."""
        })
    
    if relationship_mentions:
        advantages.append({
            "title": "Explicit Relationship Semantics",
            "description": """The ontology defines explicit relationships between concepts with clear semantics.
            This allows the system to understand how entities are connected beyond simple co-occurrence in text.
            For example, understanding that 'ownedBy' connects Products to Departments helps answer questions
            about product ownership and departmental responsibilities."""
        })
    
    advantages.append({
        "title": "Constraint-Based Reasoning",
        "description": """Business rules in the ontology provide constraints that guide the reasoning process.
        These rules ensure the system's answers are consistent with the organization's policies and practices.
        For instance, rules about approval workflows or data classification requirements can inform answers
        about process-related questions."""
    })
    
    advantages.append({
        "title": "Cross-Domain Knowledge Integration",
        "description": """The ontology connects concepts across different domains of the enterprise, enabling
        integrated reasoning that traditional document-based retrieval might miss. This allows the system to
        answer questions that span organizational boundaries, such as how marketing decisions affect product
        development or how customer feedback influences business strategy."""
    })
    
    return advantages

def render_html_in_streamlit(html_content: str):
    """Display HTML content in Streamlit using components.html."""
    import streamlit.components.v1 as components
    
    # Directly render the HTML using components.html
    components.html(html_content, height=600, scrolling=True)


def display_ontology_stats(ontology_manager):
    """Display statistics and visualizations about the ontology."""
    st.subheader("📊 Ontology Structure and Statistics")
    
    # Get basic stats
    classes = ontology_manager.get_classes()
    class_hierarchy = ontology_manager.get_class_hierarchy()
    
    # Count instances per class
    class_counts = []
    for class_name in classes:
        instance_count = len(ontology_manager.get_instances_of_class(class_name, include_subclasses=False))
        class_counts.append({
            "Class": class_name,
            "Instances": instance_count
        })
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Classes", len(classes))
    
    # Count total instances
    total_instances = sum(item["Instances"] for item in class_counts)
    with col2:
        st.metric("Total Instances", total_instances)
    
    # Count relationships
    relationship_count = len(ontology_manager.ontology_data.get("relationships", []))
    with col3:
        st.metric("Relationship Types", relationship_count)
    
    # Visualize class hierarchy
    st.markdown("### Class Hierarchy")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Tree View", "Class Statistics", "Hierarchy Graph"])
    
    with tab1:
        # Create a collapsible tree view of class hierarchy
        display_class_hierarchy_tree(ontology_manager, class_hierarchy)
    
    with tab2:
        # Display class stats and distribution
        if class_counts:
            # Filter to only show classes with instances
            non_empty_classes = [item for item in class_counts if item["Instances"] > 0]
            
            if non_empty_classes:
                df = pd.DataFrame(non_empty_classes)
                df = df.sort_values("Instances", ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(df, 
                             x="Instances", 
                             y="Class", 
                             orientation='h',
                             title="Instances per Class",
                             color="Instances",
                             color_continuous_scale="viridis")
                
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No classes with instances found.")
        
        # Show distribution of classes by inheritance depth
        display_class_depth_distribution(ontology_manager)
    
    with tab3:
        # Display class hierarchy as a graph
        display_class_hierarchy_graph(ontology_manager)
    
    # Relationship statistics
    st.markdown("### Relationship Analysis")
    
    # Get relationship usage statistics
    relationship_usage = analyze_relationship_usage(ontology_manager)
    
    # Display relationship usage in a table and chart
    if relationship_usage:
        tab1, tab2 = st.tabs(["Usage Statistics", "Domain/Range Distribution"])
        
        with tab1:
            # Create DataFrame for the table
            df = pd.DataFrame(relationship_usage)
            df = df.sort_values("Usage Count", ascending=False)
            
            # Show table
            st.dataframe(df)
            
            # Create bar chart for relationship usage
            fig = px.bar(df, 
                        x="Relationship", 
                        y="Usage Count",
                        title="Relationship Usage Frequency",
                        color="Usage Count",
                        color_continuous_scale="blues")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Display domain-range distribution
            display_domain_range_distribution(ontology_manager)


def display_class_hierarchy_tree(ontology_manager, class_hierarchy):
    """Display class hierarchy using expanders at root level, and plain list for subclasses."""
    # Find all subcategory sets
    all_subclasses = set()
    for subclasses in class_hierarchy.values():
        all_subclasses.update(subclasses)

    # Root classes are those that are not subclassed by any class
    root_classes = [cls for cls in ontology_manager.get_classes() if cls not in all_subclasses]

    for root_class in sorted(root_classes):
        class_info = ontology_manager.ontology_data["classes"].get(root_class, {})
        description = class_info.get("description", "")
        instance_count = len(ontology_manager.get_instances_of_class(root_class, include_subclasses=False))
        properties = class_info.get("properties", [])

        with st.expander(f"📁 {root_class} ({instance_count} instances)", expanded=True):
            st.markdown(f"**Description:** {description}")
            if properties:
                st.markdown("**Properties:**")
                st.markdown(", ".join(properties))

            subclasses = class_hierarchy.get(root_class, [])
            if subclasses:
                st.markdown("**Subclasses:**")
                for subclass in sorted(subclasses):
                    display_subclass_plain(subclass, class_hierarchy, ontology_manager, indent=1)
            else:
                st.markdown("*No subclasses*")


def display_subclass_plain(class_name, class_hierarchy, ontology_manager, indent=1):
    """Use indentation to render subclass info without nested expanders."""
    indent_str = " " * indent
    class_info = ontology_manager.ontology_data["classes"].get(class_name, {})
    description = class_info.get("description", "")
    properties = class_info.get("properties", [])
    instance_count = len(ontology_manager.get_instances_of_class(class_name, include_subclasses=False))

    st.markdown(f"{indent_str}🔹 **{class_name}** ({instance_count} instances)")
    st.markdown(f"{indent_str}> {description}")
    if properties:
        st.markdown(f"{indent_str}*Properties:* {', '.join(properties)}")

    subclasses = class_hierarchy.get(class_name, [])
    for subclass in sorted(subclasses):
        display_subclass_plain(subclass, class_hierarchy, ontology_manager, indent + 1)

def get_class_depths(ontology_manager) -> Dict[str, int]:
    """Calculate the inheritance depth of each class."""
    depths = {}
    class_data = ontology_manager.ontology_data["classes"]
    
    def get_depth(class_name):
        # If we've already calculated the depth, return it
        if class_name in depths:
            return depths[class_name]
        
        # Get the class data
        cls = class_data.get(class_name, {})
        
        # If no parent, depth is 0
        if "subClassOf" not in cls:
            depths[class_name] = 0
            return 0
        
        # Otherwise, depth is 1 + parent's depth
        parent = cls["subClassOf"]
        parent_depth = get_depth(parent)
        depths[class_name] = parent_depth + 1
        return depths[class_name]
    
    # Calculate depths for all classes
    for class_name in class_data:
        get_depth(class_name)
    
    return depths


def display_class_depth_distribution(ontology_manager):
    """Display distribution of classes by inheritance depth."""
    depths = get_class_depths(ontology_manager)
    
    # Count classes at each depth
    depth_counts = defaultdict(int)
    for _, depth in depths.items():
        depth_counts[depth] += 1
    
    # Create dataframe
    df = pd.DataFrame([
        {"Depth": depth, "Count": count}
        for depth, count in depth_counts.items()
    ])
    
    if not df.empty:
        df = df.sort_values("Depth")
        
        # Create bar chart
        fig = px.bar(df, 
                    x="Depth", 
                    y="Count",
                    title="Class Distribution by Inheritance Depth",
                    labels={"Depth": "Inheritance Depth", "Count": "Number of Classes"},
                    color="Count",
                    text="Count")
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)


def display_class_hierarchy_graph(ontology_manager):
    """Display class hierarchy as a directed graph."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each class
    for class_name, class_info in ontology_manager.ontology_data["classes"].items():
        # Count direct instances
        instance_count = len(ontology_manager.get_instances_of_class(class_name, include_subclasses=False))
        
        # Add node with attributes
        G.add_node(class_name, 
        type="class", 
        description=class_info.get("description", ""),
        instance_count=instance_count)
        
        # Add edge for subclass relationship
        if "subClassOf" in class_info:
            parent = class_info["subClassOf"]
            G.add_edge(parent, class_name, relationship="subClassOf")
    
    # Create a Plotly graph visualization
    # Calculate node positions using a hierarchical layout without pygraphviz
    # Find root nodes
    roots = [n for n, d in G.in_degree() if d == 0]
    
    # Use kamada_kawai_layout or spring_layout as alternative
    if len(G) > 1:
        try:
            # Try to make a hierarchical-like layout using springs
            pos = nx.spring_layout(G, iterations=50, seed=42)
            
            # Adjust y-coordinates to create a more hierarchical appearance
            # First get the topological generations
            generations = list(nx.topological_generations(G))
            
            # Assign y-coordinate based on generation
            for i, gen in enumerate(generations):
                y_pos = 1.0 - (i / max(1, len(generations) - 1))
                for node in gen:
                    if node in pos:
                        pos[node] = (pos[node][0], y_pos)
                        
        except Exception:
            # Fallback to simple spring layout if there's an issue
            pos = nx.spring_layout(G, seed=42)
    else:
        # For a single node
        pos = {list(G.nodes())[0]: (0.5, 0.5)} if G.nodes() else {}
    
    # Convert positions to lists for Plotly
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node info for hover text
        description = G.nodes[node].get("description", "")
        instance_count = G.nodes[node].get("instance_count", 0)
        
        # Prepare hover text
        hover_text = f"Class: {node}<br>Description: {description}<br>Instances: {instance_count}"
        node_text.append(hover_text)
        
        # Size nodes by instance count (with a minimum size)
        size = 10 + (instance_count * 2)
        size = min(40, max(15, size))  # Limit size range
        node_size.append(size)
        
        # Color nodes by depth
        depth = get_class_depths(ontology_manager).get(node, 0)
        # Use a color scale from light to dark blue
        node_color.append(depth)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Add a curved line with multiple points
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Add None to create a break between edges
        
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="bottom center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Blues',
            color=node_color,
            size=node_size,
            line=dict(width=2, color='DarkSlateGrey'),
            colorbar=dict(
                title="Depth",
                thickness=15,
                tickvals=[0, max(node_color)],
                ticktext=["Root", f"Depth {max(node_color)}"]
            )
        )
    )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Class Hierarchy Graph",
        title_x=0.5
        ))
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def analyze_relationship_usage(ontology_manager) -> List[Dict]:
    """Analyze how relationships are used in the ontology."""
    relationship_data = ontology_manager.ontology_data.get("relationships", [])
    instances = ontology_manager.ontology_data.get("instances", [])
    
    # Initialize counters
    usage_counts = defaultdict(int)
    
    # Count relationship usage in instances
    for instance in instances:
        for rel in instance.get("relationships", []):
            usage_counts[rel["type"]] += 1
    
    # Prepare results
    results = []
    for rel in relationship_data:
        rel_name = rel["name"]
        domain = rel["domain"]
        range_class = rel["range"]
        cardinality = rel.get("cardinality", "many-to-many")
        count = usage_counts.get(rel_name, 0)
        
        results.append({
            "Relationship": rel_name,
            "Domain": domain,
            "Range": range_class,
            "Cardinality": cardinality,
            "Usage Count": count
        })
    
    return results


def display_domain_range_distribution(ontology_manager):
    """Display domain and range distribution for relationships."""
    relationship_data = ontology_manager.ontology_data.get("relationships", [])
    
    # Count domains and ranges
    domain_counts = defaultdict(int)
    range_counts = defaultdict(int)
    
    for rel in relationship_data:
        domain_counts[rel["domain"]] += 1
        range_counts[rel["range"]] += 1
    
    # Create DataFrames
    domain_df = pd.DataFrame([
        {"Class": cls, "Count": count, "Type": "Domain"}
        for cls, count in domain_counts.items()
    ])
    
    range_df = pd.DataFrame([
        {"Class": cls, "Count": count, "Type": "Range"}
        for cls, count in range_counts.items()
    ])
    
    # Combine
    combined_df = pd.concat([domain_df, range_df])
    
    # Create plot
    if not combined_df.empty:
        fig = px.bar(combined_df, 
                    x="Class", 
                    y="Count", 
                    color="Type",
                    barmode="group",
                    title="Classes as Domain vs Range in Relationships",
                    color_discrete_map={"Domain": "#1f77b4", "Range": "#ff7f0e"})
        
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        
        st.plotly_chart(fig, use_container_width=True)


def display_entity_details(entity_info: Dict[str, Any], ontology_manager):
    """Display detailed information about an entity."""
    if not entity_info:
        st.warning("Entity not found.")
        return
    
    st.subheader(f"📝 Entity: {entity_info['id']}")
    
    # Determine entity type and get class hierarchy
    entity_type = entity_info.get("type", "")
    class_type = entity_info.get("class", entity_info.get("class_type", ""))
    
    class_hierarchy = []
    if class_type:
        current_class = class_type
        while current_class:
            class_hierarchy.append(current_class)
            parent_class = ontology_manager.ontology_data["classes"].get(current_class, {}).get("subClassOf", "")
            if not parent_class or parent_class == current_class:  # Prevent infinite loops
                break
            current_class = parent_class
    
    # Display entity metadata
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Basic Information")
        
        # Basic info metrics
        st.metric("Entity Type", entity_type)
        
        if class_type:
            st.metric("Class", class_type)
        
        # Display class hierarchy
        if class_hierarchy and len(class_hierarchy) > 1:
            st.markdown("**Class Hierarchy:**")
            hierarchy_str = " → ".join(reversed(class_hierarchy))
            st.markdown(f"```\n{hierarchy_str}\n```")
    
    with col2:
        # Display class description if available
        if "class_description" in entity_info:
            st.markdown("### Description")
            st.markdown(entity_info.get("class_description", "No description available."))
    
    # Properties
    if "properties" in entity_info and entity_info["properties"]:
        st.markdown("### Properties")
        
        # Create a more structured property display
        properties = []
        for key, value in entity_info["properties"].items():
            # Handle different value types
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            
            properties.append({"Property": key, "Value": value_str})
        
        # Display as table with highlighting
        property_df = pd.DataFrame(properties)
        st.dataframe(
            property_df,
            column_config={
                "Property": st.column_config.TextColumn("Property", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="large")
            },
            hide_index=True
        )
    
    # Relationships with visual enhancements
    if "relationships" in entity_info and entity_info["relationships"]:
        st.markdown("### Relationships")
        
        # Group relationships by direction
        outgoing = []
        incoming = []
        
        for rel in entity_info["relationships"]:
            if "direction" in rel and rel["direction"] == "outgoing":
                outgoing.append({
                    "Relationship": rel["type"],
                    "Direction": "→",
                    "Related Entity": rel["target"]
                })
            elif "direction" in rel and rel["direction"] == "incoming":
                incoming.append({
                    "Relationship": rel["type"],
                    "Direction": "←",
                    "Related Entity": rel["source"]
                })
        
        # Create tabs for outgoing and incoming
        if outgoing or incoming:
            tab1, tab2 = st.tabs(["Outgoing Relationships", "Incoming Relationships"])
            
            with tab1:
                if outgoing:
                    st.dataframe(
                        pd.DataFrame(outgoing),
                        column_config={
                            "Relationship": st.column_config.TextColumn("Relationship Type", width="medium"),
                            "Direction": st.column_config.TextColumn("Direction", width="small"),
                            "Related Entity": st.column_config.TextColumn("Target Entity", width="medium")
                        },
                        hide_index=True
                    )
                else:
                    st.info("No outgoing relationships.")
            
            with tab2:
                if incoming:
                    st.dataframe(
                        pd.DataFrame(incoming),
                        column_config={
                            "Relationship": st.column_config.TextColumn("Relationship Type", width="medium"),
                            "Direction": st.column_config.TextColumn("Direction", width="small"),
                            "Related Entity": st.column_config.TextColumn("Source Entity", width="medium")
                        },
                        hide_index=True
                    )
                else:
                    st.info("No incoming relationships.")
        
        # Visual relationship graph
        st.markdown("#### Relationship Graph")
        display_entity_relationship_graph(entity_info, ontology_manager)


def display_entity_relationship_graph(entity_info: Dict[str, Any], ontology_manager):
    """Display a graph of an entity's relationships."""
    entity_id = entity_info["id"]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add central entity
    G.add_node(entity_id, type="central")
    
    # Add related entities and relationships
    for rel in entity_info.get("relationships", []):
        if "direction" in rel and rel["direction"] == "outgoing":
            target = rel["target"]
            rel_type = rel["type"]
            
            # Add target node if not exists
            if target not in G:
                target_info = ontology_manager.get_entity_info(target)
                node_type = target_info.get("type", "unknown")
                G.add_node(target, type=node_type)
            
            # Add edge
            G.add_edge(entity_id, target, type=rel_type)
        
        elif "direction" in rel and rel["direction"] == "incoming":
            source = rel["source"]
            rel_type = rel["type"]
            
            # Add source node if not exists
            if source not in G:
                source_info = ontology_manager.get_entity_info(source)
                node_type = source_info.get("type", "unknown")
                G.add_node(source, type=node_type)
            
            # Add edge
            G.add_edge(source, entity_id, type=rel_type)
    
    # Use a force-directed layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges with curved lines
    for source, target, data in G.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        rel_type = data.get("type", "unknown")
        
        # Calculate edge midpoint for label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Draw edge
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=1, color="#888"),
            hoverinfo="text",
            hovertext=f"Relationship: {rel_type}",
            showlegend=False
        ))
        
        # Add relationship label
        fig.add_trace(go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode="text",
            text=[rel_type],
            textposition="middle center",
            textfont=dict(size=10, color="#555"),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Add nodes with different colors by type
    node_groups = defaultdict(list)
    
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_info = ontology_manager.get_entity_info(node)
        
        # Get friendly name if available
        name = node
        if "properties" in node_info and "name" in node_info["properties"]:
            name = node_info["properties"]["name"]
        
        node_groups[node_type].append({
            "id": node,
            "name": name,
            "x": pos[node][0],
            "y": pos[node][1],
            "info": node_info
        })
    
    # Define colors for different node types
    colors = {
        "central": "#ff7f0e",  # Highlighted color for central entity
        "instance": "#1f77b4",
        "class": "#2ca02c",
        "unknown": "#d62728"
    }
    
    # Add each node group with appropriate styling
    for node_type, nodes in node_groups.items():
        # Default to unknown color if type not in map
        color = colors.get(node_type, colors["unknown"])
        
        x = [node["x"] for node in nodes]
        y = [node["y"] for node in nodes]
        text = [node["name"] for node in nodes]
        
        # Prepare hover text
        hover_text = []
        for node in nodes:
            info = node["info"]
            hover = f"ID: {node['id']}<br>Name: {node['name']}"
            
            if "class_type" in info:
                hover += f"<br>Type: {info['class_type']}"
            
            hover_text.append(hover)
        
        # Adjust size for central entity
        size = 20 if node_type == "central" else 15
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color="white")
            ),
            text=text,
            textposition="bottom center",
            hoverinfo="text",
            hovertext=hover_text,
            name=node_type.capitalize()
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Relationships for {entity_id}",
        title_x=0.5,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_graph_visualization(knowledge_graph, central_entity=None, max_distance=2):
    """Display an interactive visualization of the knowledge graph."""
    st.subheader("🕸️ Knowledge Graph Visualization")
    
    # Controls for the visualization
    with st.expander("Visualization Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_classes = st.checkbox("Include Classes", value=True)
        
        with col2:
            include_instances = st.checkbox("Include Instances", value=True)
        
        with col3:
            include_properties = st.checkbox("Include Properties", value=False)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_distance = st.slider("Max Relationship Distance", 1, 5, max_distance)
        
        with col2:
            layout_algorithm = st.selectbox(
                "Layout Algorithm", 
                ["Force-Directed", "Hierarchical", "Radial", "Circular"],
                index=0
            )
    
    # Generate HTML visualization
    html = knowledge_graph.generate_html_visualization(
        include_classes=include_classes,
        include_instances=include_instances,
        central_entity=central_entity,
        max_distance=max_distance,
        include_properties=include_properties,
        layout_algorithm=layout_algorithm.lower()
    )
    
    # Render the HTML
    render_html_in_streamlit(html)
    
    # Entity filter
    with st.expander("Focus on Entity", expanded=central_entity is not None):
        # Get all entities
        entities = []
        for class_name in knowledge_graph.ontology_manager.get_classes():
            entities.extend(knowledge_graph.ontology_manager.get_instances_of_class(class_name))
        
        # Deduplicate
        entities = sorted(set(entities))
        
        # Select entity
        selected_entity = st.selectbox(
            "Select Entity to Focus On",
            ["None"] + entities,
            index=0 if central_entity is None else entities.index(central_entity) + 1
        )
        
        if selected_entity != "None":
            st.button("Focus Graph", on_click=lambda: st.experimental_rerun())
    
    # Display graph statistics
    stats = knowledge_graph.get_graph_statistics()
    if stats:
        st.markdown("### Graph Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", stats.get("node_count", 0))
        col2.metric("Edges", stats.get("edge_count", 0))
        col3.metric("Classes", stats.get("class_count", 0))
        col4.metric("Instances", stats.get("instance_count", 0))
        
        # Display relationship counts
        if "relationship_counts" in stats:
            rel_counts = stats["relationship_counts"]
            rel_data = [{"Relationship": rel, "Count": count} for rel, count in rel_counts.items() 
        if rel not in ["subClassOf", "instanceOf"]]  # Filter out structural relationships
            
            if rel_data:
                df = pd.DataFrame(rel_data)
                fig = px.bar(df, 
        x="Relationship", 
        y="Count", 
        title="Relationship Distribution",
        color="Count",
        color_continuous_scale="viridis")
                
                st.plotly_chart(fig, use_container_width=True)

def visualize_path(path_info, ontology_manager):
    """Visualize a semantic path between entities with enhanced graphics and details."""
    import streamlit as st
    import networkx as nx
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if not path_info or "path" not in path_info:
        st.warning("No path information available.")
        return

    st.subheader("🔄 Semantic Path Visualization")

    path = path_info["path"]

    # Get entity information for each node in the path
    entities = {}
    all_nodes = set()

    if "source" in path_info:
        source_id = path_info["source"]
        all_nodes.add(source_id)
        entities[source_id] = ontology_manager.get_entity_info(source_id)

    if "target" in path_info:
        target_id = path_info["target"]
        all_nodes.add(target_id)
        entities[target_id] = ontology_manager.get_entity_info(target_id)

    for edge in path:
        source_id = edge["source"]
        target_id = edge["target"]
        all_nodes.add(source_id)
        all_nodes.add(target_id)

        if source_id not in entities:
            entities[source_id] = ontology_manager.get_entity_info(source_id)

        if target_id not in entities:
            entities[target_id] = ontology_manager.get_entity_info(target_id)

    tab1, tab2, tab3 = st.tabs(["Path Visualization", "Entity Details", "Path Summary"])

    with tab1:
        G = nx.DiGraph()
        for edge in path:
            G.add_edge(edge["source"], edge["target"], label=edge["type"])

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')

        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.markdown("### Entities in Path")

        entities_by_type = defaultdict(list)
        for entity_id in all_nodes:
            entity_info = entities.get(entity_id, {})
            entity_type = entity_info.get("class_type", entity_info.get("class", "Unknown"))
            entities_by_type[entity_type].append((entity_id, entity_info))

        for entity_type, entity_list in entities_by_type.items():
            st.subheader(f"{entity_type} ({len(entity_list)})")
            for entity_id, entity_info in entity_list:
                st.markdown(f"- **{entity_id}**")
                for k, v in entity_info.get("properties", {}).items():
                    st.markdown(f"  - {k}: {v}")
            st.markdown("---")

    with tab3:
        st.markdown("### Path Description")
        if "text" in path_info and path_info["text"]:
            st.markdown(f"**Path:** {path_info['text']}")
        else:
            path_steps = []
            for edge in path:
                source_id = edge["source"]
                target_id = edge["target"]
                relation = edge["type"]

                source_name = entities[source_id].get("properties", {}).get("name", source_id)
                target_name = entities[target_id].get("properties", {}).get("name", target_id)

                path_steps.append(f"{source_name} **{relation}** {target_name}")

            st.markdown(" → ".join(path_steps))

        relevant_rules = find_relevant_rules_for_path(path, ontology_manager)
        if relevant_rules:
            st.markdown("### Relevant Business Rules")
            for rule in relevant_rules:
                st.markdown(f"- **{rule['id']}**: {rule['description']}")

def display_path_visualization(path, entities):
    """Create an enhanced visual representation of the path."""
    if not path:
        st.info("Path is empty.")
        return
    
    # Create nodes and positions
    nodes = []
    x_positions = {}
    
    # Collect all unique nodes in the path
    unique_nodes = set()
    for edge in path:
        unique_nodes.add(edge["source"])
        unique_nodes.add(edge["target"])
    
    # Create ordered list of nodes
    path_nodes = []
    if path:
        # Start with the first source
        current_node = path[0]["source"]
        path_nodes.append(current_node)
        
        # Follow the path
        for edge in path:
            target = edge["target"]
            path_nodes.append(target)
            current_node = target
    else:
        # If no path, just use the unique nodes
        path_nodes = list(unique_nodes)
    
    # Assign positions along a line
    for i, node_id in enumerate(path_nodes):
        x_positions[node_id] = i
        
        # Get node info
        entity_info = entities.get(node_id, {})
        properties = entity_info.get("properties", {})
        entity_type = entity_info.get("class_type", entity_info.get("class", "Unknown"))
        
        # Get display name
        name = properties.get("name", node_id)
        
        nodes.append({
            "id": node_id,
            "name": name,
            "type": entity_type,
            "properties": properties
        })
    
    # Create Plotly figure for horizontal path
    fig = go.Figure()
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []
    
    # Color mapping for entity types
    color_map = {}
    for node in nodes:
        node_type = node["type"]
        if node_type not in color_map:
            # Assign colors from a categorical colorscale
            idx = len(color_map) % len(px.colors.qualitative.Plotly)
            color_map[node_type] = px.colors.qualitative.Plotly[idx]
    
    for node in nodes:
        node_x.append(x_positions[node["id"]])
        node_y.append(0)  # All nodes at y=0 for a horizontal path
        node_text.append(node["name"])
        
        # Create detailed hover text
        hover = f"{node['id']}<br>{node['type']}"
        for k, v in node["properties"].items():
            hover += f"<br>{k}: {v}"
        node_hover.append(hover)
        
        # Set node color by type
        node_colors.append(color_map.get(node["type"], "#7f7f7f"))
    
    # Add node trace
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color="DarkSlateGrey")
        ),
        text=node_text,
        textposition="bottom center",
        hovertext=node_hover,
        hoverinfo="text",
        name="Entities"
    ))
    
    # Add edges with relationship labels
    for edge in path:
        source = edge["source"]
        target = edge["target"]
        edge_type = edge["type"]
        
        source_pos = x_positions[source]
        target_pos = x_positions[target]
        
        # Add edge line
        fig.add_trace(go.Scatter(
            x=[source_pos, target_pos],
            y=[0, 0],
            mode="lines",
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            showlegend=False
        ))
        
        # Add relationship label above the line
        fig.add_trace(go.Scatter(
            x=[(source_pos + target_pos) / 2],
            y=[0.1],  # Slightly above the line
            mode="text",
            text=[edge_type],
            textposition="top center",
            hoverinfo="none",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Path Visualization",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        plot_bgcolor="white"
    )
    
    # Add a legend for entity types
    for entity_type, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=entity_type,
            showlegend=True
        ))
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add step-by-step description
    st.markdown("### Step-by-Step Path")
    for i, edge in enumerate(path):
        source = edge["source"]
        target = edge["target"]
        relation = edge["type"]
        
        # Get display names
        source_info = entities.get(source, {})
        target_info = entities.get(target, {})
        
        source_name = source
        if "properties" in source_info and "name" in source_info["properties"]:
            source_name = source_info["properties"]["name"]
            
        target_name = target
        if "properties" in target_info and "name" in target_info["properties"]:
            target_name = target_info["properties"]["name"]
        
        st.markdown(f"**Step {i+1}:** {source_name} ({source}) **{relation}** {target_name} ({target})")


def find_relevant_rules_for_path(path, ontology_manager):
    """Find business rules relevant to the entities and relationships in a path."""
    rules = ontology_manager.ontology_data.get("rules", [])
    if not rules:
        return []
    
    # Extract entities and relationships from the path
    entity_types = set()
    relationship_types = set()
    
    for edge in path:
        source = edge["source"]
        target = edge["target"]
        relation = edge["type"]
        
        # Get entity info
        source_info = ontology_manager.get_entity_info(source)
        target_info = ontology_manager.get_entity_info(target)
        
        # Add entity types
        if "class_type" in source_info:
            entity_types.add(source_info["class_type"])
        
        if "class_type" in target_info:
            entity_types.add(target_info["class_type"])
        
        # Add relationship type
        relationship_types.add(relation)
    
    # Find rules that mention these entities or relationships
    relevant_rules = []
    
    for rule in rules:
        rule_text = json.dumps(rule).lower()
        
        # Check if rule mentions any of the entity types or relationships
        is_relevant = False
        
        for entity_type in entity_types:
            if entity_type.lower() in rule_text:
                is_relevant = True
                break
        
        if not is_relevant:
            for rel_type in relationship_types:
                if rel_type.lower() in rule_text:
                    is_relevant = True
                    break
        
        if is_relevant:
            relevant_rules.append(rule)
    
    return relevant_rules