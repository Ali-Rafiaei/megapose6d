"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
from typing import Tuple, Union

# Third Party
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class _Quaternion:
    """Minimal drop-in replacement for pin.Quaternion interface."""
    def __init__(self, xyzw: np.ndarray):
        self._q = xyzw

    def coeffs(self) -> np.ndarray:
        return self._q

    def tolist(self) -> list:
        return self._q.tolist()

    def __array__(self) -> np.ndarray:
        return self._q

class Transform:
    """A representation of a SE(3) object, replacing pinocchio's pin.SE3."""

    def __init__(
        self,
        *args: Union[
            Union[np.ndarray, torch.Tensor],           # T (4x4)
            Union[
                np.ndarray, torch.Tensor, Tuple[float, float, float, float]
            ],                                          # rotation
            Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],  # translation
        ]
    ):
        """
        - Transform(T): (4, 4) array
        - Transform(quaternion, translation), where
            quaternion: 4-array representing a xyzw quaternion,
                or a 3x3 rotation matrix
            translation: 3-array
        """
        if len(args) == 1:
            arg_T = args[0]
            if isinstance(arg_T, np.ndarray):
                assert arg_T.shape == (4, 4)
                self._R = arg_T[:3, :3].copy()
                self._t = arg_T[:3, -1].copy().reshape(3, 1)
            elif isinstance(arg_T, torch.Tensor):
                T = arg_T.detach().cpu().numpy().copy()
                self._R = T[:3, :3]
                self._t = T[:3, -1].reshape(3, 1)
            else:
                raise ValueError

        elif len(args) == 2:
            rotation, translation = args
            if isinstance(rotation, tuple):
                rotation_np = np.array(rotation)
            elif isinstance(rotation, torch.Tensor):
                rotation_np = rotation.detach().cpu().numpy().copy()
            elif isinstance(rotation, np.ndarray):
                rotation_np = rotation
            else:
                raise ValueError

            if rotation_np.size == 4:
                # xyzw quaternion — convert to rotation matrix via scipy
                quaternion_xyzw = rotation_np.flatten()
                self._R = Rotation.from_quat(quaternion_xyzw).as_matrix()
            elif rotation_np.size == 9:
                assert rotation_np.shape == (3, 3)
                self._R = rotation_np
            else:
                raise ValueError

            self._t = np.asarray(translation).reshape(3, 1)

        else:
            raise ValueError

    def __mul__(self, other: "Transform") -> "Transform":
        """Compose two transforms: T1 * T2"""
        R = self._R @ other._R
        t = self._R @ other._t + self._t
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return Transform(T)

    def inverse(self) -> "Transform":
        """Compute the inverse transform"""
        R_inv = self._R.T
        t_inv = -R_inv @ self._t
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = t_inv.reshape(3)
        return Transform(T)

    def __str__(self) -> str:
        return f"R:\n{self._R}\nt:\n{self._t.reshape(3)}"

    def toHomogeneousMatrix(self) -> np.ndarray:
        return self.matrix

    @property
    def translation(self) -> np.ndarray:
        return self._t.reshape(3)

    @property
    def quaternion(self) -> np.ndarray:
        """Returns xyzw quaternion as numpy array (replaces pin.Quaternion)"""
        return Rotation.from_matrix(self._R).as_quat()

    @property
    def matrix(self) -> np.ndarray:
        """Returns 4x4 homogeneous matrix representation"""
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3, 3] = self._t.reshape(3)
        return T

    @property
    def quaternion(self):
        return _Quaternion(Rotation.from_matrix(self._R).as_quat())
