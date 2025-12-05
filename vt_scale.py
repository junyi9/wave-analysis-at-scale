from vt import generate_virtual_trajectory as gen_VT
import numpy as np
import os
import multiprocessing as mp
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_WORKER_SF = None
_WORKER_PARAMS = {}

def _init_worker(npy_path,
                 x_start,
                 x_len,
                 t_len,
                 t_step,
                 interpolation,
                 westbound,
                 convert_units):
    """Pool initializer that loads shared state for worker processes."""

    global _WORKER_SF, _WORKER_PARAMS

    _WORKER_SF = np.load(npy_path, mmap_mode='r')
    _WORKER_PARAMS = {
        "x_start": x_start,
        "x_len": x_len,
        "t_len": t_len,
        "t_step": t_step,
        "interpolation": interpolation,
        "westbound": westbound,
        "convert_units": convert_units,
    }


def _process_t_start_worker(t_start: int):
    """Worker entry point executed inside Pool processes."""

    try:
        traj = gen_VT(
            _WORKER_SF,
            _WORKER_PARAMS["x_start"],
            t_start,
            x_len=_WORKER_PARAMS["x_len"],
            t_len=_WORKER_PARAMS["t_len"],
            t_step=_WORKER_PARAMS["t_step"],
            interpolation=_WORKER_PARAMS["interpolation"],
            westbound=_WORKER_PARAMS["westbound"],
            convert_units=_WORKER_PARAMS["convert_units"],
        )

        if traj is None or traj.size == 0:
            return None

        idx_col = np.full((traj.shape[0], 1), t_start, dtype=float)
        return np.hstack((idx_col, traj))

    except Exception as exc:
        print(f"t_start={t_start} failed: {exc}")
        return None


def process_file(file_base,
                 lane,
                 x_start=197,
                 x_len=4.0,  # miles
                 t_len=4.5 * 3600,  # seconds
                 t_step=0.5,  # simulation step in seconds
                 interpolation="bicubic",
                 westbound=True,
                 convert_units=True,
                 out_dir_base="vt_data"):
    """
    Process one 'smooth/{file_base}_{lane}_smooth.npy' file using multiprocessing.
    Produces per-t_start virtual trajectories, then combines them into a single CSV
    that includes the vehicle index (we use t_start as the vehicle index).
    Output layout: out_dir_base/file_base/lane/vt_all.csv
    """
    npy_path = os.path.join("smooth", f"{file_base}_{lane}_smooth.npy")
    sf = np.load(npy_path, mmap_mode='r')

    out_dir = os.path.join(out_dir_base, file_base, lane)
    os.makedirs(out_dir, exist_ok=True)

    # t_start values from 0 to t_max-1 every 1 index
    t_max = sf.shape[1]
    t_starts = list(range(0, t_max, 1))

    # choose number of worker processes
    procs = min(max(1, mp.cpu_count() - 1), len(t_starts))
    # print(f"Using {mp.cpu_count()} CPU cores, spawning {procs} workers for file {file_base})_{lane}")

    ctx = mp.get_context("fork")
    combined_csv = os.path.join(out_dir, "vt_all.csv")
    rows_written = 0

    with ctx.Pool(
        processes=procs,
        initializer=_init_worker,
        initargs=(
            npy_path,
            x_start,
            x_len,
            t_len,
            t_step,
            interpolation,
            westbound,
            convert_units,
        ),
    ) as pool, open(combined_csv, "w", encoding="utf-8") as out_f:
        out_f.write("vehicle_index,space,time,speed\n")

        for chunk in pool.imap(_process_t_start_worker, t_starts, chunksize=32):
            if chunk is None or chunk.size == 0:
                continue

            np.savetxt(out_f, chunk, delimiter=',', fmt="%.8f")
            rows_written += chunk.shape[0]

    if rows_written == 0:
        print("No trajectories generated.")
        return

    print(f"Saved {rows_written} rows to {combined_csv}")


if __name__ == "__main__":
    files = os.listdir('data')
    if not os.path.exists('smooth'):
        os.makedirs('smooth')
    files = [f for f in files if f.endswith('.npy')]
    files = sorted(files)
    for f in tqdm(files):
        process_file(f[:-14], f[-13:-8], out_dir_base='/data/vt_data')