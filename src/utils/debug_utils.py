# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import socket
from datetime import datetime
from torch.autograd.profiler import record_function

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=100_000
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       print(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       return


def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")