import logging
import os
import queue
import time

import jax
from multiprocessing import Queue, Process, Lock, log_to_stderr, get_logger

from train.train_brax import train_policy, ExperimentConfig, Hyperparameters


# Maximum time for RL brax training in SECONDS.
# If the process takes more to train the pipeline is killed.
MAX_BRAX_TRAINING_TIME = 3600


def train_multiple_models(xmls, *args, **kwargs):
    """Train multiple robot XMLs in parallel across available GPUs.

    Returns a list of (xml_path, fitness_score) tuples, or None on timeout.
    """
    if not xmls:
        return []

    logger = get_logger()
    if not logger.handlers:
        logger = log_to_stderr()
    logger.setLevel(logging.WARNING)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

    num_gpus = jax.device_count()
    logger.warning(f"[Main] Detected {num_gpus} devices")
    num_xmls = len(xmls)

    xml_queue = Queue(maxsize=num_xmls)
    for xml in xmls:
        xml_queue.put(xml)

    result_queue = Queue(maxsize=num_xmls)
    queue_lock = Lock()

    processes = []
    for gpu_idx in range(num_gpus):
        proc = Process(
            target=_process_task,
            args=(xml_queue, result_queue, queue_lock, gpu_idx) + args,
            kwargs=kwargs,
        )
        proc.start()
        processes.append(proc)

    logger.warning("[Main] All processes started")

    results = []
    deadline = time.time() + MAX_BRAX_TRAINING_TIME
    while len(results) < num_xmls and time.time() < deadline:
        try:
            result = result_queue.get(timeout=30.0)
            results.append(result)
            logger.warning(f"[Main] Got result ({len(results)}/{num_xmls}): {result}")
        except queue.Empty:
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                logger.error(f"[Main] All workers dead, only got {len(results)}/{num_xmls} results")
                break

    if len(results) < num_xmls and time.time() >= deadline:
        logger.error("[Main] Waiting for evaluation results timed out")
        for proc in processes:
            proc.terminate()
        return None

    logger.warning("[Main] All xmls evaluated, joining processes")

    # Only after everything from queue has been removed we can join processes
    # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
    for proc in processes:
        proc.join()

    logger.warning("[Main] All processes joined")
    return results


def _process_task(xml_queue, result_queue, lock, gpu_idx, *args, **kwargs):
    """Worker: train robot XMLs from the queue on a single GPU."""
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    logger.warning(f"[P:{gpu_idx}] Process initialized")

    while True:
        if not lock.acquire(timeout=30.0):
            logger.error(f"[P:{gpu_idx}] Lock acquisition timed out")
            return

        if xml_queue.empty():
            lock.release()
            logger.warning(f"[P:{gpu_idx}] Queue empty, exiting")
            return

        try:
            xml = xml_queue.get_nowait()
            logger.warning(f"[P:{gpu_idx}] Got xml: {xml}, starting training")
        except queue.Empty:
            lock.release()
            return

        lock.release()

        try:
            result = train_policy(
                ExperimentConfig(xml=xml, name=f'train_{xml}', **kwargs),
                Hyperparameters(),
            )
        except Exception as e:
            logger.error(f"[P:{gpu_idx}] Training failed: {e}")
            result = -1

        try:
            result_queue.put((xml, result), timeout=30.0)
        except Exception as e:
            logger.error(f"[P:{gpu_idx}] Failed to put result in queue: {e}")
