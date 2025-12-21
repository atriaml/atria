from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import ray
import tqdm
from atria_logger import get_logger

from atria_datasets.core.dataset._split_iterators import InstanceTransform
from atria_datasets.core.storage._mp._shard_writer_worker import ShardWriterWorker
from atria_datasets.core.storage._shard_writers._msgpack import DuplicateKeyError
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_types import BaseDataInstance

    from atria_datasets.core.dataset._datasets import SplitIterator

logger = get_logger(__name__)

_RAY_RUNTIME_ENV = {"env_vars": {"PYTHONPATH": ":".join(sys.path)}}


@ray.remote
class ShardWriterActor:
    def __init__(
        self,
        worker_id: int,
        split_dir: Path,
        data_model: type[BaseDataInstance],
        max_shard_size: int,
        preprocess_transform: InstanceTransform,
    ):
        shard_file_pattern = str(split_dir / f"{worker_id:06d}-%06d.msgpack")
        self.writer = ShardWriterWorker(
            data_model=data_model,
            storage_type=FileStorageType.MSGPACK,
            storage_file_pattern=shard_file_pattern,
            max_shard_size=max_shard_size,
            preprocess_transform=preprocess_transform,
        ).load()

    def write(self, sample_tuple):
        idx, sample = sample_tuple
        try:
            self.writer.write(idx, sample)
        except DuplicateKeyError:
            logger.error(f"Duplicate key at index {idx}, skipping")
        except Exception as e:
            logger.exception(f"Error writing sample at index {idx}")
            raise e
        return True

    def close(self):
        return self.writer.close()


class RayParallelSplitWriter:
    """Parallel split writer using Ray instead of multiprocessing queues."""

    def __init__(
        self,
        num_workers: int = 4,
        max_shard_size: int = 100_000,
        max_concurrent_tasks_limit: int = 128,
        max_memory_per_actor: int = 500 * 1024 * 1024,
    ):
        self.num_workers = num_workers
        self.max_shard_size = max_shard_size
        self._max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self._max_memory_per_actor = max_memory_per_actor
        self.actors = []

    def write_split(
        self,
        split_iterator: SplitIterator,
        split_dir: Path,
        max_samples: int | None = None,
    ) -> list:
        try:
            split_name = split_iterator.split.value
            logger.info(
                f"Writing split {split_name} with {self.num_workers} Ray actors..."
            )

            ray.init(
                num_cpus=self.num_workers,
                local_mode=self.num_workers == 1,
                runtime_env=_RAY_RUNTIME_ENV,
            )

            # Initialize actors
            self.actors = [
                ShardWriterActor.options(memory=self._max_memory_per_actor).remote(
                    worker_id=i,
                    split_dir=split_dir,
                    data_model=split_iterator.data_model,
                    max_shard_size=self.max_shard_size,
                    preprocess_transform=split_iterator._tf,
                )
                for i in range(self.num_workers)
            ]

            data_iterator = iter(split_iterator)
            pending_tasks = []
            total_samples_stored = 0
            actor_iterator = itertools.cycle(self.actors)

            for idx, sample in tqdm.tqdm(
                data_iterator, desc=f"Writing split {split_name}"
            ):
                actor = next(actor_iterator)
                pending_tasks.append(actor.write.remote((idx, sample)))
                total_samples_stored += 1

                # Limit number of concurrent in-flight tasks
                if len(pending_tasks) >= self._max_concurrent_tasks_limit:
                    ready_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1)
                    try:
                        ray.get(ready_tasks)
                    except Exception as e:
                        logger.exception("Error in shard writer actor task")
                        raise e

                if max_samples is not None and total_samples_stored >= max_samples:
                    break

            # Wait for remaining tasks to finish
            ray.get(pending_tasks)

            # Close all writers and gather results
            write_info = ray.get([actor.close.remote() for actor in self.actors])
            # Flatten results and finalize shards
            write_info = [x for shard in write_info for x in shard if x.nsamples > 0]
            for i, shard in enumerate(write_info):
                shard.shard = i + 1
            return write_info
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, shutting down Ray actors...")
            raise
        except Exception as e:
            logger.exception("Error during Ray parallel split writing")
            raise e
        finally:
            if ray.is_initialized():
                ray.shutdown()


def _queue_consumer_worker(
    worker_id: int,
    sample_queue: mp.Queue,
    result_queue: mp.Queue,
    split_dir: Path,
    data_model: type[BaseDataInstance],
    max_shard_size: int,
    preprocess_transform: InstanceTransform,
    get_timeout: float = 1.0,  # seconds
):
    """Worker function for queue-based parallel processing."""

    # Initialize the shard writer
    shard_file_pattern = str(split_dir / f"{worker_id:06d}-%06d.msgpack")
    try:
        writer = ShardWriterWorker(
            data_model=data_model,
            storage_type=FileStorageType.MSGPACK,
            storage_file_pattern=shard_file_pattern,
            max_shard_size=max_shard_size,
            preprocess_transform=preprocess_transform,
        ).load()
    except Exception as e:
        logger.exception(f"Worker {worker_id} failed to initialize writer")
        result_queue.put(("error", worker_id, e))
        return

    try:
        while True:
            idx = None
            try:
                # Non-blocking with timeout
                item = sample_queue.get(timeout=get_timeout)  # don't wait forever

                if item is None:  # sentinel to stop
                    break

                idx, sample = item
                writer.write(idx, sample)

            except DuplicateKeyError:
                # if a duplicate key error occurs, log and continue
                logger.error(
                    f"Worker {worker_id}: duplicate key at index {idx}, skipping"
                )
                continue
            except queue.Empty:
                # Timeout occurred; loop back to check for signals
                continue
            except KeyboardInterrupt:
                result_queue.put(("error", worker_id, KeyboardInterrupt()))
                return
            except Exception as e:
                result_queue.put(("error", worker_id, e))
                return
    finally:
        result_queue.put(("result", worker_id, writer.close()))


class ParallelSplitWriter:
    """Handles parallel writing of dataset splits using multiprocessing."""

    def __init__(
        self,
        num_processes: int = 4,
        max_shard_size: int = 100_000,
        queue_size: int = 10,
    ):
        self.num_processes = num_processes
        self.max_shard_size = max_shard_size
        self.queue_size = queue_size
        self._workers: list[mp.Process] = []
        self._input_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None

    def write_split(self, split_iterator: SplitIterator, split_dir: Path) -> list:
        """Write split data using multiple worker processes."""
        split_name = split_iterator.split.value
        self._input_queue = mp.Queue(maxsize=self.queue_size)
        self._result_queue = mp.Queue()

        try:
            self._start_workers(split_iterator, split_dir)
            write_info = self._process_data(split_iterator, split_name)
            return self._finalize_shards(write_info)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected. Stopping workers...")
            self._cleanup_workers()
            raise

        except Exception:
            self._cleanup_workers()
            raise

        finally:
            self._cleanup_workers()
            self._cleanup_queues()

    def _start_workers(self, split_iterator: SplitIterator, split_dir: Path):
        """Start worker processes."""
        self._workers = []
        for worker_id in range(self.num_processes):
            worker = mp.Process(
                target=_queue_consumer_worker,
                args=(
                    worker_id,
                    self._input_queue,
                    self._result_queue,
                    split_dir,
                    split_iterator.data_model,
                    self.max_shard_size,
                    split_iterator._tf,
                ),
            )
            worker.start()
            self._workers.append(worker)

    def _process_data(self, split_iterator: SplitIterator, split_name: str) -> list:
        """Process data items and handle worker communication."""
        assert self._input_queue is not None, "Input queue is not initialized."
        assert self._result_queue is not None, "Result queue is not initialized."

        data_iterator = iter(split_iterator)

        logger.info(
            f"Writing split {split_name} with {self.num_processes} processes..."
        )

        # Send data to workers
        for idx, sample in tqdm.tqdm(data_iterator, desc=f"Writing split {split_name}"):
            self._input_queue.put((idx, sample))
            self._check_for_worker_errors()

        # Send stop signals
        for _ in self._workers:
            self._input_queue.put(None)

        # Collect results
        return self._collect_results()

    def _check_for_worker_errors(self):
        """Check for worker errors without blocking."""
        assert self._result_queue is not None, "Result queue is not initialized."
        try:
            msg_type, worker_id, payload = self._result_queue.get_nowait()
            if msg_type == "error":
                for _ in self._workers:
                    self._input_queue.put(None)
                raise RuntimeError(f"Worker {worker_id} failed: {payload}")
        except queue.Empty:
            pass

    def _collect_results(self) -> list:
        """Collect results from all workers."""
        assert self._result_queue is not None, "Result queue is not initialized."

        write_info = []
        finished_workers = 0

        while finished_workers < self.num_processes:
            msg_type, worker_id, payload = self._result_queue.get()

            if msg_type == "result":
                write_info.extend(payload)
                finished_workers += 1
            else:
                raise RuntimeError(
                    f"Unexpected message type {msg_type} from worker {worker_id}"
                )

        return write_info

    def _cleanup_workers(self, force_terminate: bool = False):
        """Terminate and join all workers.

        Args:
            force_terminate: If True, terminate workers that do not exit within timeout.
        """
        assert self._input_queue is not None, "Input queue is not initialized."

        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=5)
                if worker.is_alive() and force_terminate:
                    logger.warning(f"Worker {worker.pid} did not exit, terminating.")
                    worker.terminate()
                    worker.join()

    def _cleanup_queues(self):
        """Clean up queue resources."""
        if self._input_queue:
            self._input_queue.cancel_join_thread()
            self._input_queue.close()
        if self._result_queue:
            self._result_queue.cancel_join_thread()
            self._result_queue.close()

    def _finalize_shards(self, write_info: list) -> list:
        """Filter and number shards."""
        write_info = [x for x in write_info if x.nsamples > 0]
        for i, shard in enumerate(write_info):
            shard.shard = i + 1
        return write_info


class SingleSplitWriter:
    """Handles single-process writing of dataset splits."""

    def __init__(self, max_shard_size: int = 100_000):
        self.max_shard_size = max_shard_size

    def write_split(self, split_iterator: SplitIterator, split_dir: Path) -> list:
        """Write split data using single process."""
        split_name = split_iterator.split.value
        data_iterator = iter(split_iterator)

        writer = ShardWriterWorker(
            data_model=split_iterator.data_model,
            storage_type=FileStorageType.MSGPACK,
            storage_file_pattern=str(split_dir / "000000-%06d.msgpack"),
            max_shard_size=self.max_shard_size,
            preprocess_transform=split_iterator._tf,
        ).load()

        for idx, sample in tqdm.tqdm(data_iterator, desc=f"Writing split {split_name}"):
            try:
                writer.write(idx, sample)
            except DuplicateKeyError:
                logger.error(f"Duplicate key at sample index {idx}. Skipping.")

        write_info = writer.close()
        return self._finalize_shards(write_info)

    def _finalize_shards(self, write_info: list) -> list:
        """Filter and number shards."""
        write_info = [x for x in write_info if x.nsamples > 0]
        for i, shard in enumerate(write_info):
            shard.shard = i + 1
        return write_info
