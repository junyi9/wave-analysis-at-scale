"""
Batch processing script for SGWCC analysis.

Processes multiple run IDs, lanes, and segmentation speeds in parallel.
Input structure: vt_data/{RUNID}/WB_L{lane_number}/vt_all.csv
"""

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from sgwcc_utils import sgwcc_process_file, sgwcc_stitch_components


def process_single_file(args):
    """Process a single run ID/lane/speed combination."""
    run_id, lane_number, seg_speed, speed_tolerance, file_root_local, file_path = args
    
    # Construct paths based on new structure
    lane_dir = f"WB_L{lane_number}"
    csv_file = "vt_all.csv"
    
    try:
        # Extract and track waves
        sgwcc_process_file(
            file_base=run_id,
            lane_file=os.path.join(lane_dir, csv_file),
            file_root_local=file_root_local,
            file_path=file_path,
            seg_speed=seg_speed,
            speed_tolerance=speed_tolerance,
        )
        
        # Identify connected components
        sgwcc_stitch_components(
            file_root_local=file_root_local,
            file_base=run_id,
            lane_file=os.path.join(lane_dir, csv_file),
            seg_speed=seg_speed,
            file_path=file_path,
        )
        
        print(f"✓ Completed: {run_id}/WB_L{lane_number} at {seg_speed} mph")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {run_id}/WB_L{lane_number} at {seg_speed} mph - {e}")
        return False


def batch_process(
    file_path="vt_data",
    file_root_local="results",
    seg_speeds=None,
    run_ids=None,
    lane_numbers=None,
    speed_tolerance=5.0,
):
    """
    Batch process multiple run IDs, lanes, and speeds.
    
    Args:
        file_path: Root directory containing vt_data/{RUNID}/WB_L{lane}/ structure.
        file_root_local: Output directory for results.
        seg_speeds: List of critical speeds to analyze (default: [15]).
        run_ids: List of run IDs to process (default: all in file_path).
        lane_numbers: List of lane numbers to process (default: [1, 2, 3, 4]).
        speed_tolerance: Speed tolerance in mph (default: 5.0).
    """
    if seg_speeds is None:
        seg_speeds = [15]
    
    if lane_numbers is None:
        lane_numbers = [1, 2, 3, 4]
    
    # Auto-detect run IDs if not specified
    if run_ids is None:
        run_ids = [
            d for d in os.listdir(file_path)
            if os.path.isdir(os.path.join(file_path, d)) and d != '.DS_Store'
        ]
        run_ids.sort()
        print(f"Auto-detected {len(run_ids)} run IDs from {file_path}/")
    
    # Build task list
    tasks = []
    for run_id in run_ids:
        run_dir = os.path.join(file_path, run_id)
        
        for lane_number in lane_numbers:
            lane_dir = os.path.join(run_dir, f"WB_L{lane_number}")
            csv_path = os.path.join(lane_dir, "vt_all.csv")
            
            # Check if the file exists before adding task
            if not os.path.exists(csv_path):
                print(f"⊗ Skipping {run_id}/WB_L{lane_number}: vt_all.csv not found")
                continue
            
            for seg_speed in seg_speeds:
                tasks.append((
                    run_id,
                    lane_number,
                    seg_speed,
                    speed_tolerance,
                    file_root_local,
                    file_path
                ))
    
    if not tasks:
        print("No valid tasks found. Check your input directory structure.")
        return
    
    print(f"Processing {len(tasks)} tasks using {cpu_count()} CPU cores...")
    print(f"Run IDs: {run_ids}")
    print(f"Lanes: {lane_numbers}")
    print(f"Speeds: {seg_speeds}")
    
    # Process in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_single_file, tasks)
    
    success_count = sum(results)
    print(f"\nBatch processing complete: {success_count}/{len(tasks)} succeeded")

import time


if __name__ == "__main__":
    # start = time.perf_counter()
    # # Example 1: Process all run IDs with default settings
    # batch_process(
    #     file_path="/data/vt_data",
    #     # file_root_local="results",
    #     file_root_local="/data/sgw_results",
    #     seg_speeds=[15],
    #     run_ids=None,  # Auto-detect all
    #     lane_numbers=[1,2,3,4],
    #     speed_tolerance=5.0,
    # )
    # end = time.perf_counter()
    # print(f"Total execution time: {end - start:.2f} seconds")
    
    # Example 2: Process specific run IDs and speeds
    batch_process(
        file_path="/data/vt_data",
        file_root_local="sgw_results",
        seg_speeds=[15],
        run_ids=["66e06d0203742ba297d3c30a","6900e8829478cdbd7a829345"],
        # run_ids=["66a3c80203742ba297d3c2c3"],
        lane_numbers=[1,2,3,4],
        speed_tolerance=5.0,
    )