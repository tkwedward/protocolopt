from ..core.potential import Potential
from ..core.types import StateSpace, ControlVector
from typing import Tuple, Optional, Dict, Any
import torch
import numpy as np

class QuarticPotential(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2."""

    def __init__(self, compile_mode: bool = True):
        super().__init__(compile_mode)
        self.hparams = {
            'name': self.__class__.__name__,
            'compile_mode': self.compile_mode
        }

    def potential_value(self, space_grid: StateSpace, protocol_tensor: ControlVector) -> torch.Tensor:
        """Computes the quartic potential value.

        Args:
            space_grid: Spatial coordinates.
                        Shape: (Batch, Spatial_Dim) or (Spatial_Dim,)
            protocol_tensor: Control vector [a, b].
                             Shape: (Control_Dim,)

        Returns:
            Potential value.
        """
        return torch.sum(protocol_tensor[0] * space_grid**4 - protocol_tensor[1] * space_grid**2, dim=-1)

class QuarticPotentialWithLinearTerm(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2 + c(t)x."""

    def __init__(self, compile_mode: bool = True):
        super().__init__(compile_mode)
        self.hparams = {
            'name': self.__class__.__name__,
            'compile_mode': self.compile_mode
        }

    def potential_value(self, space_grid: StateSpace, protocol_tensor: ControlVector) -> torch.Tensor:
        """Computes the quartic potential with linear term.

        Args:
            space_grid: Spatial coordinates.
                        Shape: (Batch, Spatial_Dim) or (Spatial_Dim,)
            protocol_tensor: Control vector [a, b, c].
                             Shape: (Control_Dim,)

        Returns:
            Potential value.
        """
        return torch.sum(protocol_tensor[0] * space_grid**4 - protocol_tensor[1] * space_grid**2 + protocol_tensor[2] * space_grid, dim=-1)


class QFP_Potential(Potential):
    """Potential of form V(x, t) = 1/2 * (phi_1 - phi_1x)**2 + beta * cos(phi_1) * cos(phi_1xdc/2)."""

    def __init__(self, compile_mode: bool = True, U_0 = 1.0):
        super().__init__(compile_mode)
        self.U_0 = U_0
        self.hparams = {
            'name': self.__class__.__name__,
            'compile_mode': self.compile_mode
        }

    def potential_value(self, space_grid: StateSpace, protocol_tensor: ControlVector) -> torch.Tensor:
        """Computes the quartic potential value.

        Args:
            space_grid: Spatial coordinates.
                        Shape: (Batch, Spatial_Dim) or (Spatial_Dim,)
            protocol_tensor: Control vector [a, b].
                             Shape: (Control_Dim,)

        Returns:
            Potential value.
        """
        beta = 2.3
        return torch.sum(self.U_0 * 1/2 * (space_grid - protocol_tensor[0])**2 + self.U_0 * beta * torch.cos(space_grid) * torch.cos(protocol_tensor[1]/2), dim=-1)
        # return torch.sum(1/2 * (space_grid - protocol_tensor[0])**2 + beta * torch.cos(space_grid) * torch.cos(protocol_tensor[1]/2), dim=-1)