"""
Adel-Lite: Automated Data Elements Linking - Lite

A Python library for automated schema generation and data profiling.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .schema import schema
from .profile import profile
from .map import map_relationships
from .constraints import detect_constraints
from .sample import sample
from .visualize import visualize
from .export import export_schema
from .meta import build_meta

__all__ = [
    "schema",
    "profile", 
    "map_relationships",
    "detect_constraints",
    "sample",
    "visualize",
    "export_schema",
    "build_meta",
]
