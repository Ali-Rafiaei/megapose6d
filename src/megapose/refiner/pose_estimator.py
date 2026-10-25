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
import os
# setting the MEGAPOSE_DATA_DIR environment variable
import megapose
os.environ["MEGAPOSE_DATA_DIR"] = os.path.dirname(os.path.abspath(megapose.__file__))
from collections import defaultdict
from typing import Optional, Tuple
from pathlib import Path
# from __future__ import annotations

# Third Party
import numpy as np
import torch
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# MegaPose
from megapose.refiner.pose_rigid import PosePredictor
from megapose.models import torchvision_resnet as models
import megapose.utils.tensor_collection as tc
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.lib3d.rigid_mesh_database import MeshDataBase, BatchedMeshes
from megapose.inference.types import (
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.training.utils import CudaTimer, SimpleTimer
from megapose.inference.icp_refiner import ICPRefiner


class PoseEstimator(torch.nn.Module):
    """Performs inference for pose estimation."""

    def __init__(
        self,
        refiner_model: Optional[torch.nn.Module] = None,
        depth_refiner: Optional[ICPRefiner] = None,
        bsz_objects: int = 8,
        bsz_images: int = 256,
    ) -> None:

        super().__init__()
        self.refiner_model = refiner_model
        self.bsz_objects = bsz_objects
        self.bsz_images = bsz_images


        # load cfg and mesh_db from refiner model
        if self.refiner_model is not None:
            self.cfg = self.refiner_model.cfg
            self.mesh_db = self.refiner_model.mesh_db

        self.depth_refiner = depth_refiner

        self.eval()

        self.keep_all_outputs = False

    @torch.no_grad()
    def forward_refiner(
        self,
        observation: ObservationTensor,
        data_TCO_input: PoseEstimatesType,
        n_iterations: int = 5,
        keep_all_outputs: bool = False,
        cuda_timer: bool = False,
        **refiner_kwargs,
    ) -> Tuple[dict, dict]:
        """Runs the refiner model for the specified number of iterations.


        Will actually use the batched_model_predictions to stay within
        batch size limit.

        Returns:
            (preds, extra_data)

            preds:
                A dict with keys 'refiner/iteration={n}' for
                n=1,...,n_iterations

                Each value is a data_TCO_type.

            extra_data:
                A dict containing additional information such as timing

        """

        B = data_TCO_input.poses.shape[0]
        ids = torch.arange(B)
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)
        device = observation.images.device

        preds = defaultdict(list)
        all_outputs = []
        
        for (batch_idx, (batch_ids,)) in enumerate(dl):
            data_TCO_input_ = data_TCO_input[batch_ids]
            df_ = data_TCO_input_.infos
            TCO_input_ = data_TCO_input_.poses

            # Add some additional fields to df_
            df_["refiner_batch_idx"] = batch_idx
            df_["refiner_instance_idx"] = np.arange(len(df_))

            # if df_.get("batch_im_id", None) is None:

            labels_ = df_["label"].tolist()
            # batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            # images_ = observation.images[batch_im_ids_]
            # K_ = observation.K[batch_im_ids_]
            n_batch = len(batch_ids)
            images_ = observation.images[[0]].expand(n_batch, -1, -1, -1)
            K_ = observation.K[[0]].expand(n_batch, -1, -1)

            timer_ = CudaTimer(enabled=cuda_timer)
            timer_.start()

            outputs_ = self.refiner_model(
                images=images_,
                K=K_,
                # TCO_input_ is [b,4,4] and b is B * M
                TCO=TCO_input_,
                n_iterations=n_iterations,
                labels=labels_,
                **refiner_kwargs,
            )
            # timer_.stop()
            # model_time += timer_.elapsed()

            if keep_all_outputs:
                all_outputs.append(outputs_)

            # Collect the data
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs_[f"iteration={n}"]

                infos = df_
                batch_preds = PandasTensorCollection(
                    infos,
                    poses=iter_outputs.TCO_output,
                    poses_input=iter_outputs.TCO_input,
                    K_crop=iter_outputs.K_crop,
                    K=iter_outputs.K,
                    boxes_rend=iter_outputs.boxes_rend,
                    boxes_crop=iter_outputs.boxes_crop,
                )

                preds[f"iteration={n}"].append(batch_preds)

        for k, v in preds.items():
            preds[k] = tc.concatenate(v)


        extra_data = {
            "n_iterations": n_iterations,
            "outputs": all_outputs,
            "model_time": None,
        }

        return preds, extra_data

    @torch.no_grad()
    def run_ref_pipeline(
        self,
        observation: ObservationTensor,
        initial_pose: np.ndarray,
        n_iterations: int = 5,
    ) -> PoseEstimatesType:

        n_refiner_iterations: int = n_iterations

        preds, _ = self.forward_refiner(
            observation,
            initial_pose,
            n_iterations=n_refiner_iterations,
            keep_all_outputs=False,
            cuda_timer=False,
        )
        data_TCO_refined = preds[f"iteration={n_refiner_iterations}"]

        return data_TCO_refined, preds


    def run_depth_refiner(
        self,
        observation: ObservationTensor,
        predictions: PoseEstimatesType,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs the ICP depth refiner."""
        assert self.depth_refiner is not None, "You must specify a depth refiner"
        depth = observation.depth
        K = observation.K

        refined_preds = self.depth_refiner.refine_poses_single(predictions, depth=depth, K=K)

        return refined_preds


def load_named_model(
    object_dataset: RigidObjectDataset,
) -> PoseEstimator:


    model = {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgbd-288182519",
        "requires_depth": True,
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 1,
        },
    }

    # building the path using working directory
    refiner_run_dir = Path(
            os.path.dirname(os.path.abspath(__file__)),
            "refiner_model_ckpt/refiner-rgbd-288182519/")
    cfg_path = refiner_run_dir / "config.yaml"

    refiner_cfg = yaml.load(cfg_path.read_text(), Loader=yaml.UnsafeLoader)
    if isinstance(refiner_cfg, dict):
        refiner_cfg = OmegaConf.load(cfg_path)

    refiner_cfg = check_update_config(refiner_cfg)

    mesh_db = MeshDataBase.from_object_ds(object_dataset)

    refiner_renderer = make_renderer(object_dataset)

    # print(refiner_renderer._object_dataset[0].label)
    # exit()

    mesh_db_batched = mesh_db.batched().cuda()

    run_dir = refiner_run_dir

    model = create_model_pose(refiner_cfg, renderer=refiner_renderer, mesh_db=mesh_db_batched)
    ckpt = torch.load(run_dir / "checkpoint.pth.tar")
    ckpt = ckpt["state_dict"]
    ckpt = change_keys_of_older_models(ckpt)
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = refiner_cfg
    model.config = refiner_cfg

    refiner_model = model

    depth_refiner = ICPRefiner(
        mesh_db,
        refiner_model.renderer,
    )

    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        depth_refiner=depth_refiner,
        bsz_objects=8,
        bsz_images=128,
    )
    return pose_estimator


def load_rgb_model(object_dataset: RigidObjectDataset) -> PoseEstimator:
    """Load RGB-only MegaPose refiner model (refiner-rgb-653307694).

    Used as fallback when depth is unavailable. No ICP depth refiner attached.
    """
    refiner_run_dir = Path(
            os.path.dirname(os.path.abspath(__file__)),
            "refiner_model_ckpt/refiner-rgb-653307694/"
    )
    cfg_path = refiner_run_dir / "config.yaml"

    refiner_cfg = yaml.load(cfg_path.read_text(), Loader=yaml.UnsafeLoader)
    if isinstance(refiner_cfg, dict):
        refiner_cfg = OmegaConf.load(cfg_path)

    refiner_cfg = check_update_config(refiner_cfg)

    mesh_db = MeshDataBase.from_object_ds(object_dataset)
    refiner_renderer = make_renderer(object_dataset)
    mesh_db_batched = mesh_db.batched().cuda()

    model = create_model_pose(refiner_cfg, renderer=refiner_renderer, mesh_db=mesh_db_batched)
    ckpt = torch.load(refiner_run_dir / "checkpoint.pth.tar")
    ckpt = ckpt["state_dict"]
    ckpt = change_keys_of_older_models(ckpt)
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = refiner_cfg
    model.config = refiner_cfg

    pose_estimator = PoseEstimator(
        refiner_model=model,
        depth_refiner=None,   # ICP requires depth — not available in RGB-only mode
        bsz_objects=8,
        bsz_images=128,
    )
    return pose_estimator


def make_renderer(object_dataset) -> Panda3dBatchRenderer:
    renderer_kwargs_ = {
        "preload_cache": False,
        "split_objects": False,
        "n_workers": 1,
    }

    renderer_kwargs_.setdefault("split_objects", True)
    renderer_kwargs_.setdefault("preload_cache", False)
    renderer_kwargs_.setdefault("n_workers", 4)

    renderer = Panda3dBatchRenderer(object_dataset=object_dataset, **renderer_kwargs_)

    return renderer

def check_update_config(cfg):
    """Useful for loading models previously trained with different configurations."""

    cfg.is_coarse_compat = False
    # Detect old coarse model definition
    if hasattr(cfg, "input_strategy") and cfg.input_strategy == "input=obs+one_render":
        cfg.is_coarse_compat = True
        cfg.n_rendered_views = 1
        cfg.multiview_type = "1view_TCO"
        cfg.predict_rendered_views_logits = True
        cfg.remove_TCO_rendering = True
        cfg.predict_pose_update = False

    if cfg.multiview_type == "front_3views":
        cfg.multiview_type = "TCO+front_3views"
    if cfg.multiview_type == "front_5views":
        cfg.multiview_type = "TCO+front_5views"
    if cfg.multiview_type == "front_1view":
        cfg.multiview_type = "TCO+front_1view"

    if not hasattr(cfg, "predict_pose_update"):
        cfg.predict_pose_update = True

    if not hasattr(cfg, "remove_TCO_rendering"):
        cfg.remove_TCO_rendering = False
    if not hasattr(cfg, "predict_rendered_views_logits"):
        cfg.predict_rendered_views_logits = False
    if "n_rendered_views" not in cfg:
        if "n_views" in cfg:
            cfg.n_rendered_views = cfg.n_views
            del cfg.n_views
        else:
            cfg.n_rendered_views = 1

    if "render_normals" not in cfg:
        cfg.render_normals = False
    if "render_depth" not in cfg:
        cfg.render_depth = False
    if "input_depth" not in cfg:
        cfg.input_depth = False
    if "multiview_type" not in cfg:
        cfg.multiview_type = "TCO"
        assert not cfg.remove_TCO_rendering
    if not "views_inplane_rotation" not in cfg:
        cfg.views_inplane_rotations = False
    if "depth_augmentation" not in cfg:
        cfg.depth_normalization_type = "tCR_scale"

    if "renderer" not in cfg:
        cfg.renderer = "panda3d"
    return cfg


def create_model_pose(
    cfg,
    renderer: Panda3dBatchRenderer,
    mesh_db: BatchedMeshes,
) -> PosePredictor:
    n_channels = 3
    n_normals_channels = 3 if cfg.render_normals else 0
    n_rendered_depth_channels = 1 if cfg.render_depth else 0
    n_depth_channels = 1 if cfg.input_depth else 0
    # Assumes that if you are rendering depth you are also
    # inputting it from the model
    n_inputs = (n_channels + n_depth_channels) + (
        (n_channels + n_normals_channels + n_rendered_depth_channels) * cfg.n_rendered_views
    )
    render_size = (240, 320)

    n_features = 512
    backbone = models.__dict__["resnet34"](num_classes=n_features, n_input_channels=n_inputs)
    backbone.n_features = n_features

    backbone.n_inputs = n_inputs
    model = PosePredictor(
        backbone=backbone,
        renderer=renderer,
        mesh_db=mesh_db,
        render_size=render_size,
        n_rendered_views=cfg.n_rendered_views,
        views_inplane_rotations=cfg.views_inplane_rotations,
        multiview_type=cfg.multiview_type,
        render_normals=cfg.render_normals,
        render_depth=cfg.render_depth,
        input_depth=cfg.input_depth,
        predict_rendered_views_logits=cfg.predict_rendered_views_logits,
        remove_TCO_rendering=cfg.remove_TCO_rendering,
        predict_pose_update=cfg.predict_pose_update,
        depth_normalization_type=cfg.depth_normalization_type,
    )
    return model

def change_keys_of_older_models(state_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if k.startswith("backbone.backbone"):
            new_k = "backbone." + k[len("backbone.backbone.") :]
        elif k.startswith("backbone.head.0."):
            new_k = "views_logits_head." + k[len("backbone.head.0.") :]
        else:
            new_k = k
        new_state_dict[new_k] = state_dict[k]
    return new_state_dict
