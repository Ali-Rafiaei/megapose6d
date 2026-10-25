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
from dataclasses import dataclass
from typing import List, Optional, Set, Union

# Third Party
import numpy as np
import torch
import torch.multiprocessing

# MegaPose
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.lib3d.transform import Transform
from megapose.lib3d.transform_ops import invert_transform_matrices
from megapose.utils.logging import get_logger

# Local Folder
from .panda3d_scene_renderer import Panda3dSceneRenderer
from .types import (
    CameraRenderingData,
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
    Resolution,
)

logger = get_logger(__name__)


def _ensure_virtual_display() -> None:
    """Start an Xvfb virtual display if DISPLAY is not set (headless cluster nodes).

    Must be called in the main process BEFORE workers are forked so that workers
    inherit the DISPLAY env var and the pandagl (X11) backend works normally.
    """
    if "DISPLAY" in os.environ:
        return

    import subprocess
    import time

    # Find a free display slot (check the X11 lock file)
    display_num = None
    for d in range(99, 110):
        if not os.path.exists(f"/tmp/.X{d}-lock"):
            display_num = d
            break
    if display_num is None:
        raise RuntimeError(
            "[Panda3D] Could not find a free X display number for Xvfb (checked :99-:109)."
        )

    try:
        proc = subprocess.Popen(
            ["Xvfb", f":{display_num}", "-screen", "0", "1024x768x24", "+extension", "GLX"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "[Panda3D] Xvfb not found on this node. Install it (apt-get install xvfb) "
            "or set the DISPLAY environment variable manually before running."
        )

    time.sleep(1.5)  # give Xvfb time to start
    if proc.poll() is not None:
        raise RuntimeError(
            f"[Panda3D] Xvfb exited immediately (returncode={proc.returncode}). "
            "Check that Xvfb is properly installed."
        )

    os.environ["DISPLAY"] = f":{display_num}"
    print(f"[Panda3D] Started Xvfb virtual display at DISPLAY={os.environ['DISPLAY']}", flush=True)


@dataclass
class RenderOutput:
    """
    rgb: (h, w, 3) uint8
    normals: (h, w, 3) uint8
    depth: (h, w, 1) float32
    """

    data_id: int
    rgb: torch.Tensor
    normals: Optional[torch.Tensor]
    depth: Optional[torch.Tensor]


@dataclass
class BatchRenderOutput:
    """
    rgb: (bsz, 3, h, w) float, values in [0, 1]
    normals: (bsz, 3, h, w) float, values in [0, 1]
    depth: (bsz, 1, h, w) float, in meters.
    """

    rgbs: torch.Tensor
    normals: Optional[torch.Tensor]
    depths: Optional[torch.Tensor]


@dataclass
class SceneData:
    camera_data: Panda3dCameraData
    light_datas: List[Panda3dLightData]
    object_datas: List[Panda3dObjectData]


@dataclass
class RenderArguments:
    data_id: int
    render_normals: bool
    render_depth: bool
    scene_data: SceneData


@dataclass
class _WorkerError:
    """Sentinel put into out_queue when a worker crashes."""
    worker_id: int
    traceback: str


@dataclass
class _WorkerReady:
    """Sentinel put into out_queue after successful worker initialization."""
    worker_id: int


def worker_loop(
    worker_id: int,
    in_queue: torch.multiprocessing.Queue,
    out_queue: torch.multiprocessing.Queue,
    object_dataset: RigidObjectDataset,
    preload_labels: Set[str] = set(),
) -> None:
    import traceback as _tb
    try:
        print(f"[Panda3D] Worker {worker_id}: starting Panda3D init...", flush=True)
        renderer = Panda3dSceneRenderer(
            asset_dataset=object_dataset,
            preload_labels=preload_labels,
        )
        print(f"[Panda3D] Worker {worker_id}: init complete, sending ready signal.", flush=True)
        out_queue.put(_WorkerReady(worker_id=worker_id))

        while True:
            render_args: Union[RenderArguments, None] = in_queue.get()
            if render_args is None:
                break

            scene_data = render_args.scene_data
            is_valid = (
                np.isfinite(scene_data.object_datas[0].TWO.toHomogeneousMatrix()).all()
                and np.isfinite(scene_data.camera_data.TWC.toHomogeneousMatrix()).all()
                and np.isfinite(scene_data.camera_data.K).all()
            )

            if is_valid:
                # Set copy_arrays=True so that the numpy
                # arrays are contiguous. This ensures that they
                # have non-negative strides and can be converted into
                # torch.tensors.
                renderings = renderer.render_scene(
                    object_datas=scene_data.object_datas,
                    camera_datas=[scene_data.camera_data],
                    light_datas=scene_data.light_datas,
                    render_normals=render_args.render_normals,
                    render_depth=render_args.render_depth,
                    copy_arrays=True,  # ensures non-negative strid
                )
                renderings_ = renderings[0]
            else:
                h, w = scene_data.camera_data.resolution
                renderings_ = CameraRenderingData(
                    rgb=np.zeros((h, w, 3), dtype=np.uint8),
                    normals=np.zeros((h, w, 1), dtype=np.uint8),
                    depth=np.zeros((h, w, 1), dtype=np.float32),
                )

            output = RenderOutput(
                data_id=render_args.data_id,
                rgb=torch.from_numpy(renderings_.rgb),
                normals=torch.from_numpy(renderings_.normals)
                if render_args.render_normals
                else None,
                depth=torch.from_numpy(renderings_.depth)
                if render_args.render_depth
                else None,
            )
            del render_args
            out_queue.put(output)

        logger.debug(f"Close worker: {worker_id}")

    except Exception:
        # Propagate the full traceback to the main process so it gets a real
        # error message instead of hanging at out_queue.get() forever.
        out_queue.put(_WorkerError(worker_id=worker_id, traceback=_tb.format_exc()))


class Panda3dBatchRenderer:
    def __init__(
        self,
        object_dataset: RigidObjectDataset,
        n_workers: int = 8,
        preload_cache: bool = True,
        split_objects: bool = False,
    ):

        assert n_workers >= 1
        self._object_dataset = object_dataset
        self._n_workers = n_workers
        self._split_objects = split_objects

        self._init_renderers(preload_cache)
        self._is_closed = False

    def make_scene_data(
        self,
        labels: List[str],
        TCO: torch.Tensor,
        K: torch.Tensor,
        light_datas: List[List[Panda3dLightData]],
        resolution: Resolution,
    ) -> List[SceneData]:
        """_summary_

        Args:
            labels (List[str]): _description_
            TCO (torch.Tensor): (bsz, 4, 4) float
            K (torch.Tensor): (bsz, 3, 3) float
            light_datas (List[List[Panda3dLightData]]): _description_
            resolution (Resolution): _description_

        Returns:
            List[SceneData]: _description_
        """
        bsz = TCO.shape[0]
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        TCO = TCO.detach()
        TOC = invert_transform_matrices(TCO).cpu().numpy().astype(np.float32)
        K = K.cpu().numpy()
        TWO = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        scene_datas = []
        for label_n, TOC_n, K_n, lights_n in zip(labels, TOC, K, light_datas):
            scene_data = SceneData(
                camera_data=Panda3dCameraData(
                    TWC=Transform(TOC_n),
                    K=K_n,
                    resolution=resolution,
                ),
                object_datas=[
                    Panda3dObjectData(
                        label=label_n,
                        TWO=TWO,
                    )
                ],
                light_datas=lights_n,
            )
            scene_datas.append(scene_data)
        return scene_datas

    def render(
        self,
        labels: List[str],
        TCO: torch.Tensor,
        K: torch.Tensor,
        light_datas: List[List[Panda3dLightData]],
        resolution: Resolution,
        render_depth: bool = False,
        render_mask: bool = False,
        render_normals: bool = False,
    ) -> BatchRenderOutput:

        if render_mask:
            raise NotImplementedError

        scene_datas = self.make_scene_data(labels, TCO, K, light_datas, resolution)
        bsz = len(scene_datas)

        for n, scene_data_n in enumerate(scene_datas):
            render_args = RenderArguments(
                data_id=n,
                scene_data=scene_data_n,
                render_depth=render_depth,
                render_normals=render_normals,
            )

            in_queue = self._object_label_to_queue[scene_data_n.object_datas[0].label]
            in_queue.put(render_args)

        list_rgbs = [None for _ in np.arange(bsz)]
        list_depths = [None for _ in np.arange(bsz)]
        list_normals = [None for _ in np.arange(bsz)]

        _RENDER_TIMEOUT = 120  # seconds; increase if objects are very large
        for n in np.arange(bsz):
            try:
                renders = self._out_queue.get(timeout=_RENDER_TIMEOUT)
            except Exception:
                raise RuntimeError(
                    f"[Panda3D] Timed out after {_RENDER_TIMEOUT}s waiting for render worker. "
                    "Worker likely crashed during Panda3D initialisation (check EGL/display setup) "
                    "or is hanging on the first render."
                )
            if isinstance(renders, _WorkerError):
                raise RuntimeError(
                    f"[Panda3D] Worker {renders.worker_id} crashed:\n{renders.traceback}"
                )
            data_id = renders.data_id
            list_rgbs[data_id] = renders.rgb
            if render_depth:
                list_depths[data_id] = renders.depth
            if render_normals:
                list_normals[data_id] = renders.normals
            del renders

        assert list_rgbs[0] is not None
        rgbs = torch.stack(list_rgbs).pin_memory().cuda(non_blocking=True)
        rgbs = rgbs.float().permute(0, 3, 1, 2) / 255

        if render_depth:
            assert list_depths[0] is not None
            depths = torch.stack(list_depths).pin_memory().cuda(non_blocking=True)
            depths = depths.float().permute(0, 3, 1, 2)
        else:
            depths = None

        if render_normals:
            assert list_normals[0] is not None
            normals = torch.stack(list_normals).pin_memory().cuda(non_blocking=True)
            normals = normals.float().permute(0, 3, 1, 2) / 255
        else:
            normals = None

        return BatchRenderOutput(
            rgbs=rgbs,
            depths=depths,
            normals=normals,
        )

    def _init_renderers(self, preload_cache: bool) -> None:
        object_labels = [obj.label for obj in self._object_dataset.list_objects]

        # Use 'spawn' (not the Linux default 'fork') so that worker processes
        # start with a clean slate and do NOT inherit the parent's GPU file
        # descriptors or CUDA driver state.  NVIDIA's EGL implementation is
        # NOT fork-safe: initialising an EGL context in a forked child that
        # inherited open /dev/nvidia* handles causes eglInitialize() to hang
        # indefinitely waiting for a driver lock.  'spawn' avoids this entirely.
        _mp = torch.multiprocessing.get_context("spawn")

        self._renderers: List[torch.multiprocessing.Process] = []
        if self._split_objects:
            self._in_queues: List[torch.multiprocessing.Queue] = [
                _mp.Queue() for _ in range(self._n_workers)
            ]
            self._worker_id_to_queue = {n: self._in_queues[n] for n in range(self._n_workers)}
            object_labels_split = np.array_split(object_labels, self._n_workers)
            self._object_label_to_queue = dict()
            for n, split in enumerate(object_labels_split):
                for label in split:
                    self._object_label_to_queue[label] = self._in_queues[n]
        else:
            object_labels_split = [object_labels for _ in range(self._n_workers)]
            self._in_queues = [_mp.Queue()]
            self._object_label_to_queue = {k: self._in_queues[0] for k in object_labels}
            self._worker_id_to_queue = {n: self._in_queues[0] for n in range(self._n_workers)}

        self._out_queue: torch.multiprocessing.Queue = _mp.Queue()

        for n in range(self._n_workers):
            if preload_cache:
                preload_labels = set(object_labels_split[n].tolist())
            else:
                preload_labels = set()
            renderer_process = _mp.Process(
                target=worker_loop,
                kwargs=dict(
                    worker_id=n,
                    in_queue=self._worker_id_to_queue[n],
                    out_queue=self._out_queue,
                    object_dataset=self._object_dataset,
                    preload_labels=preload_labels,
                ),
            )
            renderer_process.start()
            self._renderers.append(renderer_process)

        # Wait for all workers to finish Panda3D initialisation.
        # If a worker hangs (e.g. EGL can't open a window) we learn about it
        # immediately here rather than silently blocking later in render().
        _INIT_TIMEOUT = 90  # seconds — Panda3D + preloading can be slow
        ready = 0
        print(
            f"[Panda3D] Waiting for {self._n_workers} worker(s) to initialise "
            f"(timeout={_INIT_TIMEOUT}s)...",
            flush=True,
        )
        while ready < self._n_workers:
            try:
                msg = self._out_queue.get(timeout=_INIT_TIMEOUT)
            except Exception:
                raise RuntimeError(
                    f"[Panda3D] Workers failed to initialise within {_INIT_TIMEOUT}s "
                    f"({ready}/{self._n_workers} ready). "
                    "Worker likely hung during Panda3D/EGL window creation. "
                    "Check DISPLAY / EGL_VISIBLE_DEVICES / GPU driver setup."
                )
            if isinstance(msg, _WorkerError):
                raise RuntimeError(
                    f"[Panda3D] Worker {msg.worker_id} crashed during init:\n{msg.traceback}"
                )
            if isinstance(msg, _WorkerReady):
                ready += 1
                print(
                    f"[Panda3D] Worker {msg.worker_id} ready "
                    f"({ready}/{self._n_workers})",
                    flush=True,
                )
        print("[Panda3D] All workers ready.", flush=True)

    def stop(self) -> None:
        logger.debug("Stopping batch renderer...")
        if self._is_closed:
            return
        # Signal each worker to exit its while-True loop gracefully.
        for n in range(self._n_workers):
            try:
                self._worker_id_to_queue[n].put(None)
            except Exception:
                pass
        # Give workers a short window to exit on their own, then escalate.
        for renderer_process in self._renderers:
            renderer_process.join(timeout=5)
            if renderer_process.is_alive():
                renderer_process.terminate()
                renderer_process.join(timeout=3)
            if renderer_process.is_alive():
                renderer_process.kill()
                renderer_process.join(timeout=2)
        # Close queues; join_thread() flushes the background feeder thread.
        for queue in self._in_queues:
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                pass
        try:
            self._out_queue.close()
            self._out_queue.join_thread()
        except Exception:
            pass
        self._is_closed = True
        logger.debug("Batch renderer is closed.")

    def __del__(self) -> None:
        self.stop()
