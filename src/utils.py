import os
import time
import csv
from pathlib import Path

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def timer_func(func):
    """Decorator to measure execution time (seconds)."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper

def append_csv_row(filename, row):
    """Append a single row to a CSV file."""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def write_csv_header(filename, header):
    """Write header to a CSV file (overwrites if exists)."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)