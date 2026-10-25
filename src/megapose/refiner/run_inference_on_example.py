# Standard Library
import json
from pathlib import Path
from typing import List

# Third Party
import numpy as np
import pandas as pd
import torch
from megapose.refiner.pose_estimator import load_named_model, load_rgb_model

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor, PoseEstimatesType


def load_observation_tensor(
    rgb: np.ndarray,
    camera_data: dict,
    depth: np.ndarray = None,
) -> ObservationTensor:

    camera_data = CameraData.from_dict(camera_data)

    assert rgb.shape[:2] == camera_data.resolution
    if depth is not None:
        assert depth.shape[:2] == camera_data.resolution

    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)

    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def make_object_dataset(obj_model_path: Path) -> RigidObjectDataset:
    mesh_units = "mm"
    label = obj_model_path.split("/")[-1].split(".")[0]

    rigid_objects = []
    rigid_objects.append(
        RigidObject(label=label, mesh_path=obj_model_path, mesh_units=mesh_units)
    )

    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

"""
New plan: Let's have a builder class that constructs an estimator with both megapose refiner
    and ICP refiner. Then we can call each refiner separately as needed.
"""
class MegaPoseRefiner:
    def __init__(
        self,
        obj_ply_path,
        mode: str = "rgb",
    ):
        """
        Args:
            mode: "rgb"  — load RGB-only refiner model (refiner-rgb-653307694)
                  "rgbd" — load RGBD refiner model    (refiner-rgbd-288182519)
        """
        object_dataset = make_object_dataset(obj_ply_path)
        self._object_dataset = object_dataset
        self._mode = mode
        if mode == "rgbd":
            print("[MegaPose] Loading RGBD model...")
            self._estimator = load_named_model(object_dataset).cuda()
        else:
            print("[MegaPose] Loading RGB model...")
            self._estimator = load_rgb_model(object_dataset).cuda()

    def _get_estimator(self, depth):
        return self._estimator

    def run_megapose_ref(
        self,
        obj_model_path,
        rgb: np.ndarray,
        camera_data: dict,
        estimated_pose: np.ndarray,
        depth: np.ndarray = None,
        n_iterations: int = 5,
    ) -> None:

        # estimated pose should be in PoseEstimatesType format and the key
        #   "poses" should be of shape (N, 4, 4). Here we have only one object
        #   so we will expand the dimensions accordingly.
        label = obj_model_path.split("/")[-1].split(".")[0]

        estimated_pose = PoseEstimatesType(
            infos = pd.DataFrame([{"label": str(label)}]),
            poses=torch.from_numpy(np.expand_dims(estimated_pose, axis=0)).float()
        ).cuda()

        observation = load_observation_tensor(rgb, camera_data, depth=depth).cuda()

        output_iter5, preds = self._get_estimator(depth).run_ref_pipeline(
            observation, initial_pose=estimated_pose, n_iterations=n_iterations
        )

        return output_iter5, preds

    def run_megapose_ref_batch(
        self,
        obj_model_path,
        rgb: np.ndarray,
        camera_data: dict,
        estimated_poses: np.ndarray,
        depth: np.ndarray = None,
        n_iterations: int = 5,
    ):
        """Refine multiple pose hypotheses in a single batched call.

        Args:
            estimated_poses: (N, 4, 4) array of pose hypotheses (in meters).
            depth: optional depth image; if None, the RGB-only model is used.
        Returns:
            output_iter5, preds — same as run_megapose_ref but with N poses.
        """
        label = obj_model_path.split("/")[-1].split(".")[0]
        n_poses = len(estimated_poses)

        estimated_poses_t = PoseEstimatesType(
            infos=pd.DataFrame([{"label": str(label)}] * n_poses),
            poses=torch.from_numpy(np.array(estimated_poses)).float(),
        ).cuda()

        observation = load_observation_tensor(rgb, camera_data, depth=depth).cuda()

        output_iter5, preds = self._get_estimator(depth).run_ref_pipeline(
            observation, initial_pose=estimated_poses_t, n_iterations=n_iterations
        )

        return output_iter5, preds

    def close(self) -> None:
        """Shut down the Panda3D renderer worker processes."""
        try:
            self._estimator.refiner_model.renderer.stop()
        except Exception:
            pass

    def run_megapose_icp(
        self,
        rgb,
        depth,
        camera_data,
        estimated_pose,
        ):

        assert depth is not None, "run_megapose_icp requires depth"

        estimated_pose = PoseEstimatesType(
            infos = pd.DataFrame([{"label": "obj_000002"}]),
            poses=torch.from_numpy(np.expand_dims(estimated_pose, axis=0)).float()
        ).cuda()

        observation = load_observation_tensor(rgb, camera_data, depth=depth).cuda()

        refined_pose = self._estimator.run_depth_refiner(observation=observation, predictions=estimated_pose)

        return refined_pose
