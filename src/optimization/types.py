from typing import TypeVar, Union, Sequence
from numpy.typing import NDArray
import numpy as np

# Shared type variables
ST = TypeVar('ST')
ST_contra = TypeVar('ST_contra', contravariant=True)
OT = TypeVar('OT', bound=Union[int, float, Sequence[float]])
OT_co = TypeVar('OT_co', bound=Union[int, float, Sequence[float]], covariant=True)
OT_contra = TypeVar('OT_contra', bound=Union[int, float, Sequence[float]], contravariant=True)

# Common NumPy array alias for continuous optimizers
NDArrayFloat = NDArray[np.float64]

__all__ = [
    "ST",
    "ST_contra",
    "OT",
    "OT_co",
    "OT_contra",
    "NDArrayFloat",
]
