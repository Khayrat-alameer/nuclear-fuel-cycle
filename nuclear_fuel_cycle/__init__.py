"""
Nuclear Fuel Cycle Modeling and Simulation Package

This package provides tools for modeling and simulating the nuclear fuel cycle,
including mining, enrichment, fabrication, reactor operation, reprocessing, and waste disposal.
"""

__version__ = "0.1.0"
__author__ = "Nuclear Fuel Cycle Team"

from .enrichment import EnrichmentCascade, calculate_swu, value_function
from .reactor import ReactorModel, FuelAssembly
from .mining import MiningOperation, UraniumResource, InSituRecovery
from .fabrication import FuelFabrication
from .reprocessing import ReprocessingPlant, MOXFabrication
from .waste_disposal import InterimStorage, GeologicalRepository
from .simulation import FuelCycleSimulation, run_annual_simulation

__all__ = [
    "EnrichmentCascade",
    "calculate_swu",
    "value_function",
    "ReactorModel",
    "FuelAssembly",
    "MiningOperation",
    "UraniumResource",
    "InSituRecovery",
    "FuelFabrication",
    "ReprocessingPlant",
    "MOXFabrication",
    "InterimStorage",
    "GeologicalRepository",
    "FuelCycleSimulation",
    "run_annual_simulation",
]
