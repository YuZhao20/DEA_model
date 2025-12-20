"""
DEA (Data Envelopment Analysis) Models
Implementation based on Hosseinzadeh Lotfi et al. (2020)
Chapter 3: Basic DEA Models
Chapter 4: Advanced DEA Models
"""

from .ccr import CCRModel
from .bcc import BCCModel
from .ap import APModel
from .maj import MAJModel

__version__ = "1.0.0"
__all__ = ["CCRModel", "BCCModel", "APModel", "MAJModel"]

