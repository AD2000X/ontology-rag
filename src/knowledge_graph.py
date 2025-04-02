# src/knowledge_graph.py

import networkx as nx
from pyvis.network import Network
import json
from typing import Dict, List, Any, Optional, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

class KnowledgeGraph:
    """
    Handles the construction and visualization of knowledge graphs
    based on the ontology data.
    """
    
    def __init__(self, ontology_manager=None):
        """
        Initialize the knowledge graph handler.
        
        Args:
            ontology_manager: Optional ontology manager instance
        """
        self.ontology_manager = ontology_manager
        self.graph = None
        
        if ontology_manager:
            self.graph = ontology_manager.graph
    
    def build_visualization_graph(
        self, 
        include_classes: bool = True,
        include_instances: bool = True,
        central_entity: Optional[str] = None,
        max_distance: int = 2,
        include_properties: bool = False
    ) -> nx.Graph:
        """
        Build a simplified graph for visualization purposes.
        
        Args:
            include_classes: Whether to include class nodes
            include_instances: Whether to include instance nodes
            central_entity: Optional central entity to focus the graph on
            max_distance: Maximum distance from central entity to include
            include_properties: Whether to include property nodes
            
        Returns:
            A NetworkX graph suitable for visualization
        """
        if not self.graph:
            return nx.Graph()
        
        # Create an undirected graph for visualization
        viz_graph = nx.Graph()
        
        # If we have a central entity, extract a subgraph around it
        if central_entity and central_entity in self.graph:
            # Get nodes within max_distance of central_entity
            nodes_to_include = set([central_entity])
            current_distance = 0
            current_layer = set([central_entity])
            
            while current_distance < max_distance:
                next_layer = set()
                for node in current_layer:
                    # Get neighbors
                    neighbors = set(self.graph.successors(node)).union(set(self.graph.predecessors(node)))
                    next_layer.update(neighbors)
                
                nodes_to_include.update(next_layer)
                current_layer = next_layer
                current_distance += 1
            
            # Create subgraph
            subgraph = self.graph.subgraph(nodes_to_include)
        else:
            subgraph = self.graph
        
        # Add nodes to the visualization graph
        for node, data in subgraph.nodes(data=True):
            node_type = data.get("type")
            
            # Skip nodes based on configuration
            if node_type == "class" and not include_classes:
                continue
            if node_type == "instance" and not include_instances:
                continue
            
            # Get readable name for the node
            if node_type == "instance" and "properties" in data:
                label = data["properties"].get("name", node)
            else:
                label = node
            
            # Set node attributes for visualization
            viz_attrs = {
                "id": node,
                "label": label,
                "title": self._get_node_tooltip(node, data),
                "group": data.get("class_type", node_type),
                "shape": "dot" if node_type == "instance" else "diamond"
            }
            
            # Highlight central entity if specified
            if central_entity and node == central_entity:
                viz_attrs["color"] = "#ff7f0e"  # Orange for central entity
                viz_attrs["size"] = 25  # Larger size for central entity
            
            # Add the node
            viz_graph.add_node(node, **viz_attrs)
            
            # Add property nodes if configured
            if include_properties and node_type == "instance" and "properties" in data:
                for prop_name, prop_value in data["properties"].items():
                    # Create a property node
                    prop_node_id = f"{node}_{prop_name}"
                    prop_value_str = str(prop_value)
                    if len(prop_value_str) > 20:
                        prop_value_str = prop_value_str[:17] + "..."
                    
                    viz_graph.add_node(
                        prop_node_id, 
                        id=prop_node_id,
                        label=f"{prop_name}: {prop_value_str}",
                        title=f"{prop_name}: {prop_value}",
                        group="property",
                        shape="ellipse",
                        size=5
                    )
                    
                    # Connect instance to property
                    viz_graph.add_edge(node, prop_node_id, label="has_property", dashes=True)
        
        # Add edges to the visualization graph
        for source, target, data in subgraph.edges(data=True):
            # Only include edges between nodes that are in the viz_graph
            if source in viz_graph and target in viz_graph:
                # Skip property-related edges if we're manually creating them
                if include_properties and (
                    source.startswith(target + "_") or target.startswith(source + "_")
                ):
                    continue
                
                # Set edge attributes
                edge_type = data.get("type", "unknown")
                
                # Don't show subClassOf and instanceOf relationships if not explicitly requested
                if edge_type in ["subClassOf", "instanceOf"] and not include_classes:
                    continue
                
                viz_graph.add_edge(source, target, label=edge_type, title=edge_type)
        
        return viz_graph
    
    def _get_node_tooltip(self, node_id: str, data: Dict) -> str:
        """Generate a tooltip for a node."""
        tooltip = f"<strong>ID:</strong> {node_id}<br>"
        
        node_type = data.get("type")
        if node_type:
            tooltip += f"<strong>Type:</strong> {node_type}<br>"
        
        if node_type == "instance":
            tooltip += f"<strong>Class:</strong> {data.get('class_type', 'unknown')}<br>"
            
            # Add properties
            if "properties" in data:
                tooltip += "<strong>Properties:</strong><br>"
                for key, value in data["properties"].items():
                    tooltip += f"- {key}: {value}<br>"
        
        elif node_type == "class":
            tooltip += f"<strong>Description:</strong> {data.get('description', '')}<br>"
            
            # Add properties if available
            if "properties" in data:
                tooltip += "<strong>Properties:</strong> " + ", ".join(data["properties"]) + "<br>"
        
        return tooltip
    
    def generate_html_visualization(
        self, 
        include_classes: bool = True,
        include_instances: bool = True,
        central_entity: Optional[str] = None,
        max_distance: int = 2,
        include_properties: bool = False,
        height: str = "600px",
        width: str = "100%",
        bgcolor: str = "#ffffff",
        font_color: str = "#000000",
        layout_algorithm: str = "force-directed"
    ) -> str:
        """
        Generate an HTML visualization of the knowledge graph.
        
        Args:
            include_classes: Whether to include class nodes
            include_instances: Whether to include instance nodes
            central_entity: Optional central entity to focus the graph on
            max_distance: Maximum distance from central entity to include
            include_properties: Whether to include property nodes
            height: Height of the visualization
            width: Width of the visualization
            bgcolor: Background color
            font_color: Font color
            layout_algorithm: Algorithm for layout ('force-directed', 'hierarchical', 'radial', 'circular')
            
        Returns:
            HTML string containing the visualization
        """
        # Build the visualization graph
        viz_graph = self.build_visualization_graph(
            include_classes=include_classes,
            include_instances=include_instances,
            central_entity=central_entity,
            max_distance=max_distance,
            include_properties=include_properties
        )
        
        # Create a PyVis network
        net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color, directed=True)
        
        # Configure physics based on the selected layout algorithm
        if layout_algorithm == "force-directed":
            physics_options = {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 100
                }
            }
        elif layout_algorithm == "hierarchical":
            physics_options = {
                "enabled": True,
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 100,
                    "springConstant": 0.01,
                    "nodeDistance": 120
                },
                "solver": "hierarchicalRepulsion",
                "stabilization": {
                    "enabled": True,
                    "iterations": 100
                }
            }
            
            # Set hierarchical layout
            net.set_options("""
                var options = {
                    "layout": {
                        "hierarchical": {
                            "enabled": true,
                            "direction": "UD",
                            "sortMethod": "directed",
                            "nodeSpacing": 150,
                            "treeSpacing": 200
                        }
                    }
                }
            """)
        elif layout_algorithm == "radial":
            physics_options = {
                "enabled": True,
                "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 120,
                    "centralGravity": 0.2,
                    "springLength": 200,
                    "springConstant": 0.05
                },
                "stabilization": {
                    "enabled": True,
                    "iterations": 100
                }
            }
        elif layout_algorithm == "circular":
            physics_options = {
                "enabled": False
            }
            
            # Compute circular layout and set fixed positions
            pos = nx.circular_layout(viz_graph)
            for node_id, coords in pos.items():
                if node_id in viz_graph.nodes:
                    x, y = coords
                    viz_graph.nodes[node_id]['x'] = float(x) * 500
                    viz_graph.nodes[node_id]['y'] = float(y) * 500
                    viz_graph.nodes[node_id]['physics'] = False
        
        # Configure other options
        options = {
            "nodes": {
                "font": {"size": 12},
                "scaling": {"min": 10, "max": 30}
            },
            "edges": {
                "color": {"inherit": True},
                "smooth": {"enabled": True, "type": "dynamic"},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "font": {"size": 10, "align": "middle"}
            },
            "physics": physics_options,
            "interaction": {
                "hover": True,
                "navigationButtons": True,
                "keyboard": True,
                "tooltipDelay": 100
            }
        }
        
        # Set options and create the network
        net.options = options
        net.from_nx(viz_graph)
        
        # Add custom CSS for better visualization
        custom_css = """
        <style>
          .vis-network {
            border: 1px solid #ddd;
            border-radius: 5px;
          }
          .vis-tooltip {
            position: absolute;
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            color: #333;
            max-width: 300px;
            z-index: 9999;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
        </style>
        """
        
        # Generate the HTML and add custom CSS
        html = net.generate_html()
        html = html.replace("<style>", custom_css + "<style>")
        
        # Add legend
        legend_html = self._generate_legend_html(viz_graph)
        html = html.replace("</body>", legend_html + "</body>")
        
        return html
    
    def _generate_legend_html(self, graph: nx.Graph) -> str:
        """Generate a legend for the visualization."""
        # Collect unique groups
        groups = set()
        for _, attrs in graph.nodes(data=True):
            if "group" in attrs and attrs["group"] is not None:
                groups.add(attrs["group"])
        
        # 過濾並排序groups，確保沒有None值
        sorted_groups = sorted([g for g in groups if g is not None])
        
        # Generate HTML for legend
        legend_html = """
        <div id="graph-legend" style="position: absolute; top: 10px; right: 10px; background-color: rgba(255,255,255,0.8); 
                                    padding: 10px; border-radius: 5px; border: 1px solid #ddd; max-width: 200px;">
            <strong>Legend:</strong>
            <ul style="list-style-type: none; padding-left: 0; margin-top: 5px;">
        """
        
        # Add items for each group
        for group in sorted_groups:
            color = "#97c2fc"  # Default color
            if group == "property":
                color = "#ffcc99"
            elif group == "class":
                color = "#a1d3a2"
            
            legend_html += f"""
                <li style="margin-bottom: 5px;">
                    <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                                background-color: {color}; margin-right: 5px;"></span>
                    {group}
                </li>
            """
        
        # Close the legend container
        legend_html += """
            </ul>
            <div style="font-size: 10px; margin-top: 5px; color: #666;">
                Double-click to zoom, drag to pan, scroll to zoom in/out
            </div>
        </div>
        """
        
        return legend_html
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about the knowledge graph.
        
        Returns:
            A dictionary containing graph statistics
        """
        if not self.graph:
            return {}
        
        # Count nodes by type
        class_count = 0
        instance_count = 0
        property_count = 0
        
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type")
            if node_type == "class":
                class_count += 1
            elif node_type == "instance":
                instance_count += 1
                if "properties" in data:
                    property_count += len(data["properties"])
        
        # Count edges by type
        relationship_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "unknown")
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        # Calculate graph metrics
        try:
            # Some metrics only work on undirected graphs
            undirected = nx.Graph(self.graph)
            avg_degree = sum(dict(undirected.degree()).values()) / undirected.number_of_nodes()
            
            # Only calculate these if the graph is connected
            if nx.is_connected(undirected):
                avg_path_length = nx.average_shortest_path_length(undirected)
                diameter = nx.diameter(undirected)
            else:
                # Get the largest connected component
                largest_cc = max(nx.connected_components(undirected), key=len)
                largest_cc_subgraph = undirected.subgraph(largest_cc)
                
                avg_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
                diameter = nx.diameter(largest_cc_subgraph)
                
            # Calculate density
            density = nx.density(self.graph)
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(undirected)
        except:
            avg_degree = 0
            avg_path_length = 0
            diameter = 0
            density = 0
            clustering = 0
        
        # Count different entity types
        class_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            if data.get("type") == "instance":
                class_type = data.get("class_type", "unknown")
                class_counts[class_type] += 1
        
        # Get nodes with highest centrality
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            degree = nx.degree_centrality(self.graph)
            
            # Get top 5 nodes by betweenness centrality
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
            
            central_nodes = {
                "betweenness": [{"node": node, "centrality": round(cent, 3)} for node, cent in top_betweenness],
                "degree": [{"node": node, "centrality": round(cent, 3)} for node, cent in top_degree]
            }
        except:
            central_nodes = {}
        
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "class_count": class_count,
            "instance_count": instance_count,
            "property_count": property_count,
            "relationship_counts": relationship_counts,
            "class_instance_counts": dict(class_counts),
            "average_degree": avg_degree,
            "average_path_length": avg_path_length,
            "diameter": diameter,
            "density": density,
            "clustering_coefficient": clustering,
            "central_nodes": central_nodes
        }
    
    def find_paths_between_entities(
        self, 
        source_entity: str, 
        target_entity: str, 
        max_length: int = 3
    ) -> List[List[Dict]]:
        """
        Find all paths between two entities up to a maximum length.
        
        Args:
            source_entity: Starting entity ID
            target_entity: Target entity ID
            max_length: Maximum path length
            
        Returns:
            A list of paths, where each path is a list of edge dictionaries
        """
        if not self.graph or source_entity not in self.graph or target_entity not in self.graph:
            return []
        
        # Use networkx to find simple paths
        try:
            simple_paths = list(nx.all_simple_paths(
                self.graph, source_entity, target_entity, cutoff=max_length
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        
        # Convert paths to edge sequences
        paths = []
        for path in simple_paths:
            edge_sequence = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                # There may be multiple edges between nodes
                edges = self.graph.get_edge_data(source, target)
                if edges:
                    for key, data in edges.items():
                        edge_sequence.append({
                            "source": source,
                            "target": target,
                            "type": data.get("type", "unknown")
                        })
            
            # Only include the path if it has meaningful relationships
            # Filter out paths that only contain structural relationships like subClassOf, instanceOf
            meaningful_relationships = [edge for edge in edge_sequence 
                                      if edge["type"] not in ["subClassOf", "instanceOf"]]
            
            if meaningful_relationships:
                paths.append(edge_sequence)
        
        # Sort paths by length (shorter paths first)
        paths.sort(key=len)
        
        return paths
    
    def get_entity_neighborhood(
        self, 
        entity_id: str, 
        max_distance: int = 1,
        include_classes: bool = True
    ) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity.
        
        Args:
            entity_id: The central entity ID
            max_distance: Maximum distance from the central entity
            include_classes: Whether to include class relationships
            
        Returns:
            A dictionary containing the neighborhood information
        """
        if not self.graph or entity_id not in self.graph:
            return {}
        
        # Get nodes within max_distance of entity_id using BFS
        nodes_at_distance = {0: [entity_id]}
        visited = set([entity_id])
        
        for distance in range(1, max_distance + 1):
            nodes_at_distance[distance] = []
            
            for node in nodes_at_distance[distance - 1]:
                # Get neighbors
                neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
                
                for neighbor in neighbors:
                    # Skip class nodes if not including classes
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if not include_classes and neighbor_data.get("type") == "class":
                        continue
                    
                    if neighbor not in visited:
                        nodes_at_distance[distance].append(neighbor)
                        visited.add(neighbor)
        
        # Flatten the nodes
        all_nodes = [node for nodes in nodes_at_distance.values() for node in nodes]
        
        # Extract the subgraph
        subgraph = self.graph.subgraph(all_nodes)
        
        # Build neighbor information
        neighbors = []
        for node in all_nodes:
            if node == entity_id:
                continue
                
            node_data = self.graph.nodes[node]
            
            # Determine the relations to central entity
            relations = []
            
            # Check direct relationships
            # Check if central entity is source
            edges_out = self.graph.get_edge_data(entity_id, node)
            if edges_out:
                for key, data in edges_out.items():
                    rel_type = data.get("type", "unknown")
                    
                    # Skip structural relationships if not including classes
                    if not include_classes and rel_type in ["subClassOf", "instanceOf"]:
                        continue
                    
                    relations.append({
                        "type": rel_type,
                        "direction": "outgoing"
                    })
            
            # Check if central entity is target
            edges_in = self.graph.get_edge_data(node, entity_id)
            if edges_in:
                for key, data in edges_in.items():
                    rel_type = data.get("type", "unknown")
                    
                    # Skip structural relationships if not including classes
                    if not include_classes and rel_type in ["subClassOf", "instanceOf"]:
                        continue
                    
                    relations.append({
                        "type": rel_type,
                        "direction": "incoming"
                    })
            
            # Also find paths through intermediate nodes (indirect relationships)
            if not relations:  # Only look for indirect if no direct relationships
                for path_length in range(2, max_distance + 1):
                    try:
                        # Find paths of exactly length path_length
                        paths = list(nx.all_simple_paths(
                            self.graph, entity_id, node, cutoff=path_length, min_edges=path_length
                        ))
                        
                        for path in paths:
                            if len(path) > 1:  # Path should have at least 2 nodes
                                intermediate_nodes = path[1:-1]  # Skip source and target
                                
                                # Format the path as a relation
                                path_relation = {
                                    "type": "indirect_connection",
                                    "direction": "outgoing",
                                    "path_length": len(path) - 1,
                                    "intermediates": intermediate_nodes
                                }
                                
                                relations.append(path_relation)
                                
                                # Only need one example of an indirect path
                                break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            
            # Only include neighbors with relations
            if relations:
                neighbors.append({
                    "id": node,
                    "type": node_data.get("type"),
                    "class_type": node_data.get("class_type"),
                    "properties": node_data.get("properties", {}),
                    "relations": relations,
                    "distance": next(dist for dist, nodes in nodes_at_distance.items() if node in nodes)
                })
        
        # Group neighbors by distance
        neighbors_by_distance = defaultdict(list)
        for neighbor in neighbors:
            neighbors_by_distance[neighbor["distance"]].append(neighbor)
        
        # Get central entity info
        central_data = self.graph.nodes[entity_id]
        
        return {
            "central_entity": {
                "id": entity_id,
                "type": central_data.get("type"),
                "class_type": central_data.get("class_type", ""),
                "properties": central_data.get("properties", {})
            },
            "neighbors": neighbors,
            "neighbors_by_distance": dict(neighbors_by_distance),
            "total_neighbors": len(neighbors)
        }
    
    def find_common_patterns(self) -> List[Dict[str, Any]]:
        """
        Find common patterns and structures in the knowledge graph.
        
        Returns:
            A list of pattern dictionaries
        """
        if not self.graph:
            return []
        
        patterns = []
        
        # Find common relationship patterns
        relationship_patterns = self._find_relationship_patterns()
        if relationship_patterns:
            patterns.extend(relationship_patterns)
        
        # Find hub entities (entities with many connections)
        hub_entities = self._find_hub_entities()
        if hub_entities:
            patterns.append({
                "type": "hub_entities",
                "description": "Entities with high connectivity serving as knowledge hubs",
                "entities": hub_entities
            })
        
        # Find common property patterns
        property_patterns = self._find_property_patterns()
        if property_patterns:
            patterns.extend(property_patterns)
        
        return patterns
    
    def _find_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Find common relationship patterns in the graph."""
        # Count relationship triplets (source_type, relation, target_type)
        triplet_counts = defaultdict(int)
        
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("type", "unknown")
            
            # Skip structural relationships
            if rel_type in ["subClassOf", "instanceOf"]:
                continue
            
            # Get node types
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            source_type = (
                source_data.get("class_type")
                if source_data.get("type") == "instance"
                else source_data.get("type")
            )
            
            target_type = (
                target_data.get("class_type")
                if target_data.get("type") == "instance"
                else target_data.get("type")
            )
            
            if source_type and target_type:
                triplet = (source_type, rel_type, target_type)
                triplet_counts[triplet] += 1
        
        # Get patterns with significant frequency (more than 1 occurrence)
        patterns = []
        for triplet, count in triplet_counts.items():
            if count > 1:
                source_type, rel_type, target_type = triplet
                
                # Find examples of this pattern
                examples = []
                for source, target, data in self.graph.edges(data=True):
                    if len(examples) >= 3:  # Limit to 3 examples
                        break
                        
                    rel = data.get("type", "unknown")
                    if rel != rel_type:
                        continue
                    
                    source_data = self.graph.nodes[source]
                    target_data = self.graph.nodes[target]
                    
                    current_source_type = (
                        source_data.get("class_type")
                        if source_data.get("type") == "instance"
                        else source_data.get("type")
                    )
                    
                    current_target_type = (
                        target_data.get("class_type")
                        if target_data.get("type") == "instance"
                        else target_data.get("type")
                    )
                    
                    if current_source_type == source_type and current_target_type == target_type:
                        # Get readable names if available
                        source_name = source
                        if source_data.get("type") == "instance" and "properties" in source_data:
                            properties = source_data["properties"]
                            if "name" in properties:
                                source_name = properties["name"]
                        
                        target_name = target
                        if target_data.get("type") == "instance" and "properties" in target_data:
                            properties = target_data["properties"]
                            if "name" in properties:
                                target_name = properties["name"]
                        
                        examples.append({
                            "source": source,
                            "source_name": source_name,
                            "target": target,
                            "target_name": target_name,
                            "relationship": rel_type
                        })
                
                patterns.append({
                    "type": "relationship_pattern",
                    "description": f"{source_type} {rel_type} {target_type}",
                    "source_type": source_type,
                    "relationship": rel_type,
                    "target_type": target_type,
                    "count": count,
                    "examples": examples
                })
                
                patterns.sort(key=lambda x: x["count"], reverse=True)
        
        return patterns
    
    def _find_hub_entities(self) -> List[Dict[str, Any]]:
            """Find entities that serve as hubs (many connections)."""
            # Calculate degree centrality
            degree = nx.degree_centrality(self.graph)
            
            # Get top entities by degree
            top_entities = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
            
            hub_entities = []
            for node, centrality in top_entities:
                node_data = self.graph.nodes[node]
                node_type = node_data.get("type")
                
                # Only consider instance nodes
                if node_type == "instance":
                    # Get class type
                    class_type = node_data.get("class_type", "unknown")
                    
                    # Get name if available
                    name = node
                    if "properties" in node_data and "name" in node_data["properties"]:
                        name = node_data["properties"]["name"]
                    
                    # Count relationships by type
                    relationships = defaultdict(int)
                    for _, _, data in self.graph.edges(data=True, nbunch=[node]):
                        rel_type = data.get("type", "unknown")
                        if rel_type not in ["subClassOf", "instanceOf"]:
                            relationships[rel_type] += 1
                    
                    hub_entities.append({
                        "id": node,
                        "name": name,
                        "type": class_type,
                        "centrality": centrality,
                        "relationships": dict(relationships),
                        "total_connections": sum(relationships.values())
                    })
            
            # Sort by total connections
            hub_entities.sort(key=lambda x: x["total_connections"], reverse=True)
            
            return hub_entities
        
    def _find_property_patterns(self) -> List[Dict[str, Any]]:
        """Find common property patterns in instance data."""
        # Track properties by class type
        properties_by_class = defaultdict(lambda: defaultdict(int))
        
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "instance":
                class_type = data.get("class_type", "unknown")
                
                if "properties" in data:
                    for prop in data["properties"].keys():
                        properties_by_class[class_type][prop] += 1
        
        # Find common property combinations
        patterns = []
        for class_type, props in properties_by_class.items():
            # Sort properties by frequency
            sorted_props = sorted(props.items(), key=lambda x: x[1], reverse=True)
            
            # Only include classes with multiple instances
            class_instances = sum(1 for _, data in self.graph.nodes(data=True)
                                if data.get("type") == "instance" and data.get("class_type") == class_type)
            
            if class_instances > 1:
                common_props = [prop for prop, count in sorted_props if count > 1]
                
                if common_props:
                    patterns.append({
                        "type": "property_pattern",
                        "description": f"Common properties for {class_type} instances",
                        "class_type": class_type,
                        "instance_count": class_instances,
                        "common_properties": common_props,
                        "property_frequencies": {prop: count for prop, count in sorted_props}
                    })
        
        return patterns