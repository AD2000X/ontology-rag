# src/ontology_manager.py

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Set

class OntologyManager:
    """
    Manages the ontology model and provides methods for querying and navigating
    the ontological structure.
    """
    
    def __init__(self, ontology_path: str):
        """
        Initialize the ontology manager with a path to the ontology JSON file.
        
        Args:
            ontology_path: Path to the JSON file containing the ontology model
        """
        self.ontology_path = ontology_path
        self.ontology_data = self._load_ontology()
        self.graph = nx.MultiDiGraph()
        self._build_graph()
        
    def _load_ontology(self) -> Dict:
        """Load the ontology from the JSON file."""
        with open(self.ontology_path, 'r') as f:
            return json.load(f)
    
    def _build_graph(self):
        """Build the ontology graph from the JSON data."""
        # Add classes
        for class_id, class_data in self.ontology_data["classes"].items():
            self.graph.add_node(
                class_id,
                type="class",
                description=class_data.get("description", ""),
                properties=class_data.get("properties", [])
            )
    
            # Handle subclass relations
            if "subClassOf" in class_data:
                parent = class_data["subClassOf"]
                self.graph.add_edge(class_id, parent, type="subClassOf")
    
        # Add relationships (schema-level only, no edge added yet)
        for rel in self.ontology_data.get("relationships", []):
            pass  # schema relationships are used for metadata, not edges
    
        # Add instances
        for instance in self.ontology_data.get("instances", []):
            instance_id = instance["id"]
            class_type = instance["type"]
            properties = instance.get("properties", {})
    
            # Add the instance node
            self.graph.add_node(
                instance_id,
                type="instance",
                class_type=class_type,
                properties=properties
            )
    
            # Link instance to its class
            self.graph.add_edge(instance_id, class_type, type="instanceOf")
    
            # Add relationship edges if any
            for rel in instance.get("relationships", []):
                target = rel.get("target")
                rel_type = rel.get("type")
                if target and rel_type:
                    self.graph.add_edge(instance_id, target, type=rel_type)

    
    def get_classes(self) -> List[str]:
        """Return a list of all class names in the ontology."""
        return list(self.ontology_data["classes"].keys())
    
    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """Return a dictionary mapping each class to its subclasses."""
        hierarchy = {}
        for class_id in self.get_classes():
            hierarchy[class_id] = []
        
        for class_id, class_data in self.ontology_data["classes"].items():
            if "subClassOf" in class_data:
                parent = class_data["subClassOf"]
                if parent in hierarchy:
                    hierarchy[parent].append(class_id)
        
        return hierarchy
    
    def get_instances_of_class(self, class_name: str, include_subclasses: bool = True) -> List[str]:
        """
        Get all instances of a given class.
        
        Args:
            class_name: The name of the class
            include_subclasses: Whether to include instances of subclasses
            
        Returns:
            A list of instance IDs
        """
        if include_subclasses:
            # Get all subclasses recursively
            subclasses = set(self._get_all_subclasses(class_name))
            subclasses.add(class_name)
            
            # Get instances of all classes
            instances = []
            for class_id in subclasses:
                instances.extend([
                    n for n, attr in self.graph.nodes(data=True)
                    if attr.get("type") == "instance" and attr.get("class_type") == class_id
                ])
            return instances
        else:
            # Just get direct instances
            return [
                n for n, attr in self.graph.nodes(data=True)
                if attr.get("type") == "instance" and attr.get("class_type") == class_name
            ]
    
    def _get_all_subclasses(self, class_name: str) -> List[str]:
        """Recursively get all subclasses of a given class."""
        subclasses = []
        direct_subclasses = [
            src for src, dst, data in self.graph.edges(data=True)
            if dst == class_name and data.get("type") == "subClassOf"
        ]
        
        for subclass in direct_subclasses:
            subclasses.append(subclass)
            subclasses.extend(self._get_all_subclasses(subclass))
            
        return subclasses
    
    def get_relationships(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Get all relationships for a given entity, optionally filtered by type.
        
        Args:
            entity_id: The ID of the entity
            relationship_type: Optional relationship type to filter by
            
        Returns:
            A list of dictionaries containing relationship information
        """
        relationships = []
        
        # Look at outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            rel_type = data.get("type")
            if rel_type != "instanceOf" and rel_type != "subClassOf":
                if relationship_type is None or rel_type == relationship_type:
                    relationships.append({
                        "type": rel_type,
                        "target": target,
                        "direction": "outgoing"
                    })
        
        # Look at incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            rel_type = data.get("type")
            if rel_type != "instanceOf" and rel_type != "subClassOf":
                if relationship_type is None or rel_type == relationship_type:
                    relationships.append({
                        "type": rel_type,
                        "source": source,
                        "direction": "incoming"
                    })
                    
        return relationships
    
    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> List[List[Dict]]:
        """
        Find all paths between two entities up to a maximum length.
        
        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_length: Maximum path length
            
        Returns:
            A list of paths, where each path is a list of relationship dictionaries
        """
        paths = []
        
        # Use networkx to find simple paths
        simple_paths = nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length)
        
        for path in simple_paths:
            path_with_edges = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                # There may be multiple edges between nodes
                edges = self.graph.get_edge_data(source, target)
                if edges:
                    for key, data in edges.items():
                        path_with_edges.append({
                            "source": source,
                            "target": target,
                            "type": data.get("type", "unknown")
                        })
            paths.append(path_with_edges)
            
        return paths
    
    def get_entity_info(self, entity_id: str) -> Dict:
        """
        Get detailed information about an entity.
        
        Args:
            entity_id: The ID of the entity
            
        Returns:
            A dictionary with entity information
        """
        if entity_id not in self.graph:
            return {}
        
        node_data = self.graph.nodes[entity_id]
        entity_type = node_data.get("type")
        
        if entity_type == "instance":
            # Get class information
            class_type = node_data.get("class_type")
            class_info = self.ontology_data["classes"].get(class_type, {})
            
            return {
                "id": entity_id,
                "type": entity_type,
                "class": class_type,
                "class_description": class_info.get("description", ""),
                "properties": node_data.get("properties", {}),
                "relationships": self.get_relationships(entity_id)
            }
        elif entity_type == "class":
            return {
                "id": entity_id,
                "type": entity_type,
                "description": node_data.get("description", ""),
                "properties": node_data.get("properties", []),
                "subclasses": self._get_all_subclasses(entity_id),
                "instances": self.get_instances_of_class(entity_id)
            }
        
        return node_data
    
    def get_text_representation(self) -> str:
        """
        Generate a text representation of the ontology for embedding.
        
        Returns:
            A string containing the textual representation of the ontology
        """
        text_chunks = []
        
        # Class definitions
        for class_id, class_data in self.ontology_data["classes"].items():
            chunk = f"Class: {class_id}\n"
            chunk += f"Description: {class_data.get('description', '')}\n"
            
            if "subClassOf" in class_data:
                chunk += f"{class_id} is a subclass of {class_data['subClassOf']}.\n"
            
            if "properties" in class_data:
                chunk += f"{class_id} has properties: {', '.join(class_data['properties'])}.\n"
            
            text_chunks.append(chunk)
        
        # Relationship definitions
        for rel in self.ontology_data["relationships"]:
            chunk = f"Relationship: {rel['name']}\n"
            chunk += f"Domain: {rel['domain']}, Range: {rel['range']}\n"
            chunk += f"Description: {rel.get('description', '')}\n"
            chunk += f"Cardinality: {rel.get('cardinality', 'many-to-many')}\n"
            
            if "inverse" in rel:
                chunk += f"The inverse relationship is {rel['inverse']}.\n"
            
            text_chunks.append(chunk)
        
        # Rules
        for rule in self.ontology_data.get("rules", []):
            chunk = f"Rule: {rule.get('id', '')}\n"
            chunk += f"Description: {rule.get('description', '')}\n"
            text_chunks.append(chunk)
        
        # Instance data
        for instance in self.ontology_data["instances"]:
            chunk = f"Instance: {instance['id']}\n"
            chunk += f"Type: {instance['type']}\n"
            
            # Properties
            if "properties" in instance:
                props = []
                for key, value in instance["properties"].items():
                    if isinstance(value, list):
                        props.append(f"{key}: {', '.join(str(v) for v in value)}")
                    else:
                        props.append(f"{key}: {value}")
                
                if props:
                    chunk += "Properties:\n- " + "\n- ".join(props) + "\n"
            
            # Relationships
            if "relationships" in instance:
                rels = []
                for rel in instance["relationships"]:
                    rels.append(f"{rel['type']} {rel['target']}")
                
                if rels:
                    chunk += "Relationships:\n- " + "\n- ".join(rels) + "\n"
            
            text_chunks.append(chunk)
        
        return "\n\n".join(text_chunks)
    
    def query_by_relationship(self, source_type: str, relationship: str, target_type: str) -> List[Dict]:
        """
        Query for instances connected by a specific relationship.
        
        Args:
            source_type: Type of the source entity
            relationship: Type of relationship
            target_type: Type of the target entity
            
        Returns:
            A list of matching relationship dictionaries
        """
        results = []
        
        # Get all instances of the source type
        source_instances = self.get_instances_of_class(source_type)
        
        for source_id in source_instances:
            # Get relationships of the specified type
            relationships = self.get_relationships(source_id, relationship)
            
            for rel in relationships:
                if rel["direction"] == "outgoing" and "target" in rel:
                    target_id = rel["target"]
                    target_data = self.graph.nodes[target_id]
                    
                    # Check if the target is of the right type
                    if (target_data.get("type") == "instance" and 
                        target_data.get("class_type") == target_type):
                        results.append({
                            "source": source_id,
                            "source_properties": self.graph.nodes[source_id].get("properties", {}),
                            "relationship": relationship,
                            "target": target_id,
                            "target_properties": target_data.get("properties", {})
                        })
        
        return results

    def get_semantic_context(self, query: str) -> List[str]:
        """
        Retrieve relevant semantic context from the ontology based on a query.
        
        This method identifies entities and relationships mentioned in the query
        and returns contextual information about them from the ontology.
        
        Args:
            query: The query string to analyze
            
        Returns:
            A list of text chunks providing relevant ontological context
        """
        # This is a simple implementation - a more sophisticated one would use
        # entity recognition and semantic parsing
        
        query_lower = query.lower()
        context_chunks = []
        
        # Check for class mentions
        for class_id in self.get_classes():
            if class_id.lower() in query_lower:
                # Add class information
                class_data = self.ontology_data["classes"][class_id]
                chunk = f"Class {class_id}: {class_data.get('description', '')}\n"
                
                # Add subclass information
                if "subClassOf" in class_data:
                    parent = class_data["subClassOf"]
                    chunk += f"{class_id} is a subclass of {parent}.\n"
                
                # Add property information
                if "properties" in class_data:
                    chunk += f"{class_id} has properties: {', '.join(class_data['properties'])}.\n"
                
                context_chunks.append(chunk)
                
                # Also add some instance examples
                instances = self.get_instances_of_class(class_id, include_subclasses=False)[:3]
                if instances:
                    instance_chunk = f"Examples of {class_id}:\n"
                    for inst_id in instances:
                        props = self.graph.nodes[inst_id].get("properties", {})
                        if "name" in props:
                            instance_chunk += f"- {inst_id} ({props['name']})\n"
                        else:
                            instance_chunk += f"- {inst_id}\n"
                    context_chunks.append(instance_chunk)
        
        # Check for relationship mentions
        for rel in self.ontology_data["relationships"]:
            if rel["name"].lower() in query_lower:
                chunk = f"Relationship {rel['name']}: {rel.get('description', '')}\n"
                chunk += f"This relationship connects {rel['domain']} to {rel['range']}.\n"
                
                # Add examples
                examples = self.query_by_relationship(rel['domain'], rel['name'], rel['range'])[:3]
                if examples:
                    chunk += "Examples:\n"
                    for ex in examples:
                        source_props = ex["source_properties"]
                        target_props = ex["target_properties"]
                        
                        source_name = source_props.get("name", ex["source"])
                        target_name = target_props.get("name", ex["target"])
                        
                        chunk += f"- {source_name} {rel['name']} {target_name}\n"
                
                context_chunks.append(chunk)
        
        # If we found nothing specific, add general ontology info
        if not context_chunks:
            # Add information about top-level classes
            top_classes = [c for c, data in self.ontology_data["classes"].items() 
                          if "subClassOf" not in data or data["subClassOf"] == "Entity"]
            
            if top_classes:
                chunk = "Main classes in the ontology:\n"
                for cls in top_classes:
                    desc = self.ontology_data["classes"][cls].get("description", "")
                    chunk += f"- {cls}: {desc}\n"
                context_chunks.append(chunk)
            
            # Add information about key relationships
            if self.ontology_data["relationships"]:
                chunk = "Key relationships in the ontology:\n"
                for rel in self.ontology_data["relationships"][:5]:  # Top 5 relationships
                    chunk += f"- {rel['name']}: {rel.get('description', '')}\n"
                context_chunks.append(chunk)
        
        return context_chunks