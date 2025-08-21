"""
Factory for creating group graph instances.

Provides a clean interface for instantiating group graphs without hardcoded
type checks. Enables easy extension to new group types by registering them
in the factory.
"""

from typing import List, Optional, Dict, Type, Callable
from .base import AbstractGroupGraph, UnsupportedGroupError
from .cyclic import CycleGraph
from .dihedral import DihedralGraph


class GroupGraphFactory:
    """
    Factory class for creating group graph instances.
    
    **Design Pattern:**
    Uses registry pattern to map group type strings to constructor functions.
    This eliminates hardcoded if/elif chains and enables clean extension.
    
    **Extension:**
    To add new group types:
    1. Implement AbstractGroupGraph subclass
    2. Register with GroupGraphFactory.register()
    3. Use throughout codebase via GroupGraphFactory.create()
    
    **Example:**
    ```python
    # Register new group type
    GroupGraphFactory.register("tetrahedral", TetrahedralGraph)
    
    # Use anywhere in codebase
    graph = GroupGraphFactory.create("tetrahedral", nodes)
    ```
    """
    
    # Registry mapping group type names to constructor functions
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, group_type: str, graph_class: Type[AbstractGroupGraph]):
        """
        Register a new group graph class.
        
        Args:
            group_type: String identifier for the group type
            graph_class: Class implementing AbstractGroupGraph interface
        """
        if not issubclass(graph_class, AbstractGroupGraph):
            raise TypeError(f"Graph class must inherit from AbstractGroupGraph")
        
        cls._registry[group_type] = graph_class
    
    @classmethod
    def create(cls, group_type: str, nodes: List[int], 
               generator: Optional[str] = None) -> AbstractGroupGraph:
        """
        Create a group graph instance.
        
        Args:
            group_type: Type of group ('cycle', 'dihedral', 'tetrahedral', etc.)
            nodes: List of integer node labels
            generator: Optional generator specification (group-dependent)
            
        Returns:
            Group graph instance implementing AbstractGroupGraph interface
            
        Raises:
            UnsupportedGroupError: If group_type is not registered
        """
        if group_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise UnsupportedGroupError(
                f"Group type '{group_type}' not supported. "
                f"Available types: {available_types}"
            )
        
        graph_class = cls._registry[group_type]
        
        # Handle different constructor signatures
        if group_type == "dihedral":
            if generator is None:
                generator = "r-s"  # Default generator for dihedral
            return graph_class(nodes, generator)
        else:
            # Most groups don't need generator specification
            return graph_class(nodes)
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported group types."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_supported(cls, group_type: str) -> bool:
        """Check if a group type is supported."""
        return group_type in cls._registry


# Register built-in group types
GroupGraphFactory.register("cycle", CycleGraph)
GroupGraphFactory.register("cyclic", CycleGraph)  # Alias
GroupGraphFactory.register("dihedral", DihedralGraph)


# Convenience function for backward compatibility
def create_group_graph(group_type: str, nodes: List[int], 
                      generator: Optional[str] = None) -> AbstractGroupGraph:
    """
    Convenience function to create group graphs.
    
    This function provides the same interface as the factory but with
    a more direct function call syntax.
    """
    return GroupGraphFactory.create(group_type, nodes, generator)
