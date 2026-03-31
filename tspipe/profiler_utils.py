from typing import Optional
from pathlib import Path
from threading import Thread
from queue import Queue
from collections import defaultdict, deque
import json
import pynvml as nvml
import time
import threading
import torch.cuda.nvtx as nvtx
import torch


class GpuTaskProfiler:
    def __init__(self, output_dir: str = "profiling_logs", filename: str = "gpu_task_summary.txt"):
        self.queue = Queue()
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / filename
        self.trace_file = self.output_dir / f"{self.output_file.stem}.jsonl"
        self.records = []
        self.running = True
        self.thread = Thread(target=self._thread_logger, name="ProfilerThread", daemon=True)
        self.thread.start()
        print(f"[Profiler] Logging to {filename}")

    def _append_trace_record(self, record):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log(
        self,
        task_name,
        device_id,
        batch_id,
        ubatch_id,
        partition_id,
        is_target,
        time_ms,
        mem,
        max_mem,
        start_time=None,
        gpu_util=None,
        power_w=None,
        power_limit_w=None,
        wall_ms=None,
        cuda_ms=None,
        queue_wait_ms=None,
        sync_wait_ms=None,
        injected_sleep_ms=None,
        exec_wall_ms=None,
    ):
        record = {
            "task_name": task_name,
            "device": device_id,
            "batch_id": batch_id,
            "ubatch_id": ubatch_id,
            "partition": partition_id,
            "target": is_target,
            "time_ms": time_ms,              # backward compatibility
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "queue_wait_ms": queue_wait_ms,
            "sync_wait_ms": sync_wait_ms,
            "injected_sleep_ms": injected_sleep_ms,
            "exec_wall_ms": exec_wall_ms,
            "mem_MB": mem,
            "max_mem_MB": max_mem,
            "start_time": start_time,
            "gpu_util": gpu_util,
            "power_w": power_w,
            "power_limit_w": power_limit_w,
        }
        self._append_trace_record(record)
        self.queue.put(record)

    def _thread_logger(self):
        while self.running or not self.queue.empty():
            try:
                record = self.queue.get()
                if record == "STOP":
                    break
                self.records.append(record)
            except Exception:
                continue
        self._save_summary()

    def stop(self):
        self.running = False
        self.queue.put("STOP")
        self.thread.join()

    def _save_summary(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            by_step = defaultdict(list)
            for r in self.records:
                by_step[r["batch_id"]].append(r)

            for step, records in sorted(by_step.items()):
                f.write(f"\n====== Step {step} ======\n")
                for r in records:
                    line = (
                        f"[GPU {r.get('device')}] "
                        f"Task={r.get('task_name', ''):<20} | "
                        f"Batch={r.get('batch_id')} "
                        f"UBatch={r.get('ubatch_id')} | "
                        f"Partition={r.get('partition')} "
                        f"Target={r.get('target')} | "
                    )

                    if r.get("start_time") is not None:
                        line += f"StartTime={r['start_time']:.6f} | "

                    line += (
                        f"Wall={r.get('wall_ms', 0.0):.2f} ms | "
                        f"CUDA={r.get('cuda_ms', 0.0):.2f} ms | "
                        f"ExecWall={r.get('exec_wall_ms', 0.0):.2f} ms | "
                        f"QWait={r.get('queue_wait_ms', 0.0):.2f} ms | "
                        f"SyncWait={r.get('sync_wait_ms', 0.0):.2f} ms | "
                        f"InjectedSleep={r.get('injected_sleep_ms', 0.0):.2f} ms | "
                    )

                    line += (
                        f"Mem={r.get('mem_MB', 0.0):.2f} MB | "
                        f"MaxMem={r.get('max_mem_MB', 0.0):.2f} MB"
                    )

                    if r.get("gpu_util") is not None:
                        line += f" | GpuUtil={r['gpu_util']}%"

                    if r.get("power_w") is not None:
                        power_limit = r.get("power_limit_w")
                        if power_limit is not None:
                            line += f" | Power={r['power_w']:.2f}/{power_limit:.2f}W"
                        else:
                            line += f" | Power={r['power_w']:.2f}W"

                    f.write(line + "\n")

        print(f"[Profiler] Profiling summary saved to {self.output_file}")


gpu_task_profiler_instance: Optional[GpuTaskProfiler] = None


def init_gpu_task_profiler(*args, **kwargs):
    global gpu_task_profiler_instance
    gpu_task_profiler_instance = GpuTaskProfiler(*args, **kwargs)


def stop_gpu_task_profiler():
    global gpu_task_profiler_instance
    if gpu_task_profiler_instance is not None:
        gpu_task_profiler_instance.stop()


def _consume_profile_extra(ctx):
    extra = getattr(ctx, "_profile_extra", None)
    if not isinstance(extra, dict):
        return {
            "queue_wait_ms": 0.0,
            "sync_wait_ms": 0.0,
            "injected_sleep_ms": 0.0,
        }

    out = {
        "queue_wait_ms": float(extra.get("queue_wait_ms", 0.0) or 0.0),
        "sync_wait_ms": float(extra.get("sync_wait_ms", 0.0) or 0.0),
        "injected_sleep_ms": float(extra.get("injected_sleep_ms", 0.0) or 0.0),
    }
    ctx._profile_extra = {}
    return out


def create_compute_profile_hooks(task_name, task_fn):
    def wrapped_fn(ctx, task):
        if not hasattr(wrapped_fn, "nvml_initialized"):
            nvml.nvmlInit()
            wrapped_fn.gpu_handles = [
                nvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(nvml.nvmlDeviceGetCount())
            ]
            wrapped_fn.nvml_initialized = True

        if task.batch_id is None:
            return task_fn(ctx, task)

        device_id = ctx.device_id

        # profile extras reset
        ctx._profile_extra = {}

        torch.cuda.synchronize()
        wall_start = time.time()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        try:
            result = task_fn(ctx, task)
        except Exception as e:
            print(f"[GPU {device_id}] Exception during task {task_name}: {e}")
            raise

        end_evt.record()
        torch.cuda.synchronize()

        wall_ms = (time.time() - wall_start) * 1000.0
        cuda_ms = start_evt.elapsed_time(end_evt)

        extra = _consume_profile_extra(ctx)
        queue_wait_ms = extra["queue_wait_ms"]
        sync_wait_ms = extra["sync_wait_ms"]
        injected_sleep_ms = extra["injected_sleep_ms"]

        # localization input용: 전체 wall에서 queue wait만 제외
        # injected_sleep_ms는 synthetic 실험에서 localization에 반영되어야 하므로 포함
        exec_wall_ms = max(wall_ms - queue_wait_ms, 0.0)

        mem_mb = (
            torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )
        max_mem_mb = (
            torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
            if device_id is not None else 0.0
        )

        gpu_util = None
        power_w = None
        power_limit_w = None

        if device_id is not None and device_id < len(wrapped_fn.gpu_handles):
            handle = wrapped_fn.gpu_handles[device_id]
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit_w = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        if gpu_task_profiler_instance is not None:
            gpu_task_profiler_instance.log(
                task_name=task_name,
                device_id=device_id,
                batch_id=task.batch_id,
                ubatch_id=task.ubatch_id,
                partition_id=task.partition_id,
                is_target=task.is_target,
                time_ms=exec_wall_ms,    # backward compatibility: old readers still see useful value
                wall_ms=wall_ms,
                cuda_ms=cuda_ms,
                queue_wait_ms=queue_wait_ms,
                sync_wait_ms=sync_wait_ms,
                injected_sleep_ms=injected_sleep_ms,
                exec_wall_ms=exec_wall_ms,
                mem=mem_mb,
                max_mem=max_mem_mb,
                start_time=wall_start,
                gpu_util=gpu_util,
                power_w=power_w,
                power_limit_w=power_limit_w,
            )

        return result

    return wrapped_fn


class GpuUtilSampler:
    def __init__(self, interval=0.01, maxlen=5000):
        self.interval = interval
        self.running = False
        self.data = deque(maxlen=maxlen)
        self.thread = threading.Thread(target=self._run, daemon=True)
        nvml.nvmlInit()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]

    def _run(self):
        while self.running:
            timestamp = time.time()
            for i, handle in enumerate(self.handles):
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self.data.append((timestamp, i, util.gpu, power))
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_average_util(self, device_id, start, end):
        samples = [u for t, i, u, _ in self.data if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else 0.0

    def get_average_power(self, device_id, start, end):
        samples = [p for t, i, _, p in self.data if i == device_id and start <= t <= end]
        return sum(samples) / len(samples) if samples else 0.0