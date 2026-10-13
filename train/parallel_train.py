import logging
import math
import json
import traceback
import multiprocessing
import os
import queue
import time
from pathlib import Path

# Timeout for one population's RL evaluations, in seconds.
MAX_BRAX_TRAINING_TIME = 3600


def gpu_ids():
    """Return visible CUDA identifiers without remapping an existing allocation."""
    jax_visible = os.environ.get("JAX_CUDA_VISIBLE_DEVICES", "all").strip().lower()
    if jax_visible != "all":
        raise ValueError("Unset JAX_CUDA_VISIBLE_DEVICES and select the allocation with CUDA_VISIBLE_DEVICES instead")
    if os.environ.get("JAX_PLATFORM_NAME", "").strip().lower() == "cpu":
        raise ValueError("Unset JAX_PLATFORM_NAME=cpu for GPU training")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax

    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "RoboMorph training requires a CUDA GPU. Install the cuda12 extra "
            "and run on a GPU machine; see README.md."
        ) from exc
    if not devices:
        raise RuntimeError("No CUDA GPUs are available for RoboMorph training")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    identifiers = visible.split(",") if visible is not None else [str(device.id) for device in devices]
    return [identifier.strip() for identifier in identifiers[:len(devices)]]


def train_multiple_models(xmls, *, timeout=MAX_BRAX_TRAINING_TIME, **kwargs):
    """Evaluate XMLs with one spawned worker per visible GPU.

    Return (xml_path, fitness) pairs. Raise on timeout or missing worker results;
    individual training exceptions are saved with a traceback and scored -inf.
    """
    if not xmls:
        return []
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    devices = gpu_ids()[:len(xmls)]
    logger = logging.getLogger(__name__)
    logger.info("Using %d GPU workers", len(devices))
    context = multiprocessing.get_context("spawn")
    xml_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    results = []
    deadline = time.monotonic() + timeout

    try:
        for xml in xmls:
            xml_queue.put(xml)
        # Queue.empty() is unreliable with multiprocessing's feeder thread.
        # Explicit sentinels ensure every queued design is consumed exactly once.
        for _ in devices:
            xml_queue.put(None)
        for device in devices:
            process = context.Process(
                target=_process_task,
                args=(xml_queue, result_queue, device),
                kwargs=kwargs,
            )
            process.start()
            processes.append(process)

        while len(results) < len(xmls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Training timed out after {timeout}s ({len(results)}/{len(xmls)} designs evaluated)")
            try:
                results.append(result_queue.get(timeout=min(1.0, remaining)))
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    raise RuntimeError(f"All training workers exited with only {len(results)}/{len(xmls)} results")
        return results
    finally:
        # Always reap workers, including timeout and startup failure paths.
        for process in processes:
            process.join(timeout=max(0, min(1.0, deadline - time.monotonic())))
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for task_queue in (xml_queue, result_queue):
            task_queue.cancel_join_thread()
            task_queue.close()


def _process_task(xml_queue, result_queue, device, **kwargs):
    """Set GPU visibility before importing the training stack in a fresh process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    from train.train_brax import train_policy, ExperimentConfig, Hyperparameters

    logger = logging.getLogger(__name__)
    while True:
        xml = xml_queue.get()
        if xml is None:
            return
        try:
            result = train_policy(
                ExperimentConfig(xml=xml, name=Path(xml).stem, **kwargs),
                Hyperparameters(),
            )
            if not math.isfinite(result):
                raise ValueError(f"Training returned nonfinite fitness: {result}")
        except Exception as exc:
            logger.exception("Training failed for %s on GPU %s", xml, device)
            output_dir = Path(kwargs.get("output_dir", "runs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / f"failure_{Path(xml).stem}.json"
            with failure_path.open("x") as file:
                json.dump({"xml": str(xml), "error": str(exc), "traceback": traceback.format_exc()}, file, indent=2)
            result = float("-inf")
        result_queue.put((xml, result))
