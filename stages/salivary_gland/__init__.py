"""
Sporozoite Interaction Stage Simulation

High-density 3D sporozoite simulation without flow effects.
Focuses purely on sporozoite interactions and volume effects.
"""

from .sporozoite_interactions import SporozoiteInteractionManager
from .salivary_gland_simulation import SporozoiteInteractionSimulation, SporozoiteInteractionConfigs

# Maintain backward compatibility
from .salivary_gland_simulation import SalivaryGlandSimulation, SalivaryGlandConfigs

__all__ = ['SporozoiteInteractionManager', 'SporozoiteInteractionSimulation', 'SporozoiteInteractionConfigs',
           'SalivaryGlandSimulation', 'SalivaryGlandConfigs']