"""
Utility functions for identifying stop-and-go wave connected components (SGWCC).

This module provides functions for analyzing vehicle trajectories, identifying
wave fronts/tails, and tracking their propagation patterns in traffic flow data.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

# Color palette for visualization
COLOR_PALETTE = [
    "#006400",  # darkgreen
    "#00008b",  # darkblue
    "#b03060",  # maroon3
    "#ff4500",  # orangered
    "#ffd700",  # gold
    "#7fff00",  # chartreuse
    "#00ffff",  # aqua
    "#ff00ff",  # fuchsia
    "#6495ed",  # cornflower
    "#ffdab9",  # peachpuff
]


def get_colors(c_id: int) -> str:
    """Get a color from the palette using cyclic indexing.

    Args:
        c_id: Color identifier (will be cycled through palette).

    Returns:
        Hex color code string.
    """
    return COLOR_PALETTE[c_id % len(COLOR_PALETTE)]


def decompose_trajectory_fixed(
    vt_sample: pd.DataFrame, fixed_speed: float = 30
) -> pd.DataFrame:
    """Decompose a trajectory using a fixed reference speed.

    Calculates nominal (expected) position based on fixed speed and computes
    oscillation as deviation from the nominal trajectory.

    Args:
        vt_sample: DataFrame with columns 'space', 'time', and 'speed'.
        fixed_speed: Reference speed in mph (default: 30).

    Returns:
        DataFrame with added columns: 'average_speed', 'nominal_space', 'oscillation_space'.
    """
    mean_speed = fixed_speed / 3600  # Convert mph to miles per second
    vt_sample["average_speed"] = mean_speed
    initial_space = vt_sample["space"].max()
    vt_sample["nominal_space"] = (
        - initial_space + mean_speed * (vt_sample["time"] - vt_sample["time"].min())
    )
    vt_sample["oscillation_space"] = (vt_sample["space"] - vt_sample["nominal_space"])
    return vt_sample


def sgwcc_extract_wave_points(
    data: pd.DataFrame, seg_speed: float = 15, speed_tolerance: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract wave front and tail points from vehicle trajectory data.

    Identifies local maxima (wave fronts) and minima (wave tails) in trajectory
    oscillation, filtered by speed proximity to critical speed.

    Args:
        data: DataFrame with velocity trajectory data (must contain 'vehicle_index', 'speed',
              'oscillation_space', 'time', 'space' columns).
        seg_speed: Critical segmentation speed in mph (default: 15).
        speed_tolerance: Speed tolerance threshold in mph for filtering peaks (default: 5.0).

    Returns:
        Tuple of (wave_front_df, wave_tail_df) with columns:
        ['time', 'space', 'speed', 'vehicle_index', 'wave_type'].
    
    Note:
        wave_type: 1 for wave front, 2 for wave tail.
    """
    vehicle_ids = data["vehicle_index"].unique()
    wave_front_list = []
    wave_tail_list = []

    for vehicle_id in tqdm(vehicle_ids, desc="Extracting wave points"):
        vehicle_data = data[data["vehicle_index"] == vehicle_id].copy().reset_index(drop=True)
        vehicle_data = decompose_trajectory_fixed(vehicle_data, fixed_speed=seg_speed)

        # Extract wave fronts (local maxima)
        peaks, _ = find_peaks(vehicle_data["oscillation_space"])
        for peak in peaks:
            if abs(vehicle_data["speed"][peak] - seg_speed) <= speed_tolerance:
                wave_front_list.append(
                    (
                        vehicle_data["time"][peak],
                        vehicle_data["space"][peak],
                        vehicle_data["speed"][peak],
                        vehicle_data["vehicle_index"][peak],
                        1, # wave_type 1 for front
                    )
                )

        # Extract wave tails (local minima)
        peaks, _ = find_peaks(-vehicle_data["oscillation_space"])
        for peak in peaks:
            if abs(vehicle_data["speed"][peak] - seg_speed) <= speed_tolerance:
                wave_tail_list.append(
                    (
                        vehicle_data["time"][peak],
                        vehicle_data["space"][peak],
                        vehicle_data["speed"][peak],
                        vehicle_data["vehicle_index"][peak],
                        2, # wave_type 2 for tail
                    )
                )

    wave_front = pd.DataFrame(
        wave_front_list, columns=["time", "space", "speed", "vehicle_index", "wave_type"]
    )
    wave_tail = pd.DataFrame(
        wave_tail_list, columns=["time", "space", "speed", "vehicle_index", "wave_type"]
    )

    return wave_front, wave_tail


def sgwcc_pair_wave_points(
    all_points: pd.DataFrame,
) -> pd.DataFrame:
    """Pair corresponding wave fronts and tails for each vehicle.

    Groups wave observations by vehicle and matches fronts with tails chronologically.

    Args:
        all_points: Combined DataFrame of wave points with 'vehicle_index', 'wave_type', 'time'.

    Returns:
        DataFrame with paired front/tail data. Columns are prefixed with '_front' or '_tail'.
    """
    pair_data = pd.DataFrame()
    vehicle_id_list = all_points["vehicle_index"].unique()

    for vehicle_id in vehicle_id_list:
        vehicle_data = (
            all_points[all_points["vehicle_index"] == vehicle_id].copy().reset_index(drop=True)
        )
        vehicle_data = vehicle_data.sort_values(by="time").reset_index(drop=True)

        if len(vehicle_data) <= 1:
            continue

        # Remove leading tail if present
        if vehicle_data["wave_type"].iloc[0] == 2:
            vehicle_data = vehicle_data.iloc[1:].reset_index(drop=True)
            vehicle_data = vehicle_data.sort_values(by="time", ascending=False).reset_index(
                drop=True
            )
            if len(vehicle_data) > 0 and vehicle_data["wave_type"].iloc[0] == 1:
                vehicle_data = vehicle_data.iloc[1:].reset_index(drop=True)

        # Split and align fronts/tails
        vehicle_front = (
            vehicle_data[vehicle_data["wave_type"] == 1]
            .sort_values(by="time")
            .reset_index(drop=True)
        )
        vehicle_tail = (
            vehicle_data[vehicle_data["wave_type"] == 2]
            .sort_values(by="time")
            .reset_index(drop=True)
        )

        if len(vehicle_front) > 0 and len(vehicle_tail) > 0:
            vehicle_pair_data = pd.concat([vehicle_front, vehicle_tail], axis=1).reset_index(
                drop=True
            )
            vehicle_pair_data.columns = [
                "time_front",
                "space_front",
                "speed_front",
                "v_id_front",
                "wave_type_front",
                "lane_front",
                "unique_index_front",
                "time_tail",
                "space_tail",
                "speed_tail",
                "v_id_tail",
                "wave_type_tail",
                "lane_tail",
                "unique_index_tail",
            ]
            pair_data = pd.concat([pair_data, vehicle_pair_data], axis=0).reset_index(
                drop=True
            )

    return pair_data.dropna().reset_index(drop=True)


def sgwcc_trace_wave_trajectory(
    wave_data: pd.DataFrame, window_time_ahead: float = 15, window_time_behind: float = 5, window_space_ahead: float = 0.02, window_space_behind: float = 0.05
) -> List[List]:
    """Trace wave trajectory across consecutive vehicle detections.

    Connects wave observations from consecutive vehicles based on spatio-temporal proximity.

    Args:
        wave_data: DataFrame with wave points (time, space, vehicle_index, unique_index, mask).
        window_time_ahead: Time window ahead for matching consecutive detections (seconds).
        window_time_behind: Time window behind for matching consecutive detections (seconds).
        window_space_ahead: Space window ahead for matching consecutive detections (miles).
        window_space_behind: Space window behind for matching consecutive detections (miles).

    Returns:
        List of traces, each containing [time, space, vehicle_index, trace_id, unique_id].
    """
    trace = []
    lane_wave = wave_data.copy()
    lane_wave["mask"] = 0
    all_record = len(lane_wave)
    trace_id = 0

    while len(lane_wave[lane_wave["mask"] == 0]) > 0:
        lane_wave = lane_wave[lane_wave["mask"] == 0]
        vt_list = np.sort(lane_wave["vehicle_index"].unique())

        if trace_id % 30 == 0:
            remaining = len(lane_wave[lane_wave["mask"] == 0]) / all_record * 100
            print(f"Traces found: {trace_id}, Remaining: {remaining:.1f}%")

        trace_id += 1

        # Start new trace from first untraced wave point
        vt_data = lane_wave[lane_wave["vehicle_index"] == vt_list[0]].iloc[0:1]
        lane_wave.loc[vt_data.index, "mask"] = 1

        time = vt_data["time"].values[0]
        space = vt_data["space"].values[0]
        unique_id = vt_data["unique_index"].values[0]
        trace.append([time, space, vt_list[0], trace_id, unique_id])

        vt_index_old = vt_list[0]

        # Continue trace through consecutive vehicle IDs
        for vt_index in vt_list[1:]:
            if vt_index - vt_index_old != 1:
                break

            vt_data = lane_wave[lane_wave["vehicle_index"] == vt_index]
            local_data = vt_data[
                (vt_data["time"] <= time + window_time_ahead)
                & (vt_data["time"] >= time - window_time_behind)
                & (vt_data["space"] >= space - window_space_ahead)
                & (vt_data["space"] <= space + window_space_behind)
            ].sort_values(by="time", ascending=False)

            vt_index_old = vt_index

            if len(local_data) == 0:
                break

            # Use most recent match
            local_data = local_data.iloc[0:1]
            lane_wave.loc[local_data.index, "mask"] = 1

            time = local_data["time"].values[0]
            space = local_data["space"].values[0]
            unique_id = local_data["unique_index"].values[0]
            trace.append([time, space, vt_index, trace_id, unique_id])

    return trace


def sgwcc_plot_trajectories(
    trace_df: pd.DataFrame,
    output_path: Path,
    wave_type: str = "front",
    seg_speed: int = 15,
) -> None:
    """Plot wave trajectory paths on time-space diagram.

    Args:
        trace_df: DataFrame with columns [time, space, v_id, trace_id, unique_id].
        output_path: Path to save the output PDF.
        wave_type: Type of wave ("front" or "tail") for title/labeling.
        seg_speed: Critical speed used for segmentation (for labeling).
    """
    # Avoid LaTeX dependency; use standard matplotlib text rendering
    plt.rc("text", usetex=False)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 60

    grouped = trace_df.groupby("trace_id")
    plt.figure(figsize=(45, 15))

    for _, group in grouped:
        if len(group) > 1:
            plt.plot(group["time"], group["space"], linewidth=4)

    plt.gca().invert_yaxis()
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Mile Marker")
    plt.grid(linestyle="--", lw=2, alpha=1)
    plt.xlim(0, 14400)
    plt.xticks(
        np.arange(0, 14401, 1800),
        ["6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00"],
    )
    plt.ylim(0, 4)
    plt.yticks(
        np.arange(0, 4.1, 0.5),
        ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"],
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def sgwcc_save_wave_points(
    wave_front: pd.DataFrame,
    wave_tail: pd.DataFrame,
    lane_number: int,
    seg_speed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Save wave front/tail points and return combined DataFrame.

    Args:
        wave_front: Wave front data.
        wave_tail: Wave tail data.
        lane_number: Lane identifier.
        seg_speed: Critical segmentation speed.
        output_dir: Directory for output files.

    Returns:
        Combined DataFrame with all points and unique indices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    wave_front["lane"] = lane_number
    wave_tail["lane"] = lane_number

    all_points = pd.concat([wave_front, wave_tail], axis=0).reset_index(drop=True)
    all_points["unique_index"] = all_points.index

    wave_tail_out = all_points[all_points["wave_type"] == 2].copy().reset_index(drop=True)
    wave_front_out = all_points[all_points["wave_type"] == 1].copy().reset_index(drop=True)

    wave_tail_out.to_csv(output_dir / f"wave_tail_{seg_speed}.csv", index=False)
    print(f"Saved: {output_dir / f'wave_tail_{seg_speed}.csv'}")

    wave_front_out.to_csv(output_dir / f"wave_front_{seg_speed}.csv", index=False)
    print(f"Saved: {output_dir / f'wave_front_{seg_speed}.csv'}")

    return all_points


def sgwcc_process_file(
    file_base: str,
    lane_file: str,
    file_root_local: str,
    file_path: str,
    seg_speed: int = 15,
    speed_tolerance: float = 5.0,
) -> None:
    """Main orchestration function for stop-and-go wave cluster characterization.

    Processes a vehicle trajectory file to identify, pair, and track wave fronts and tails.
    Outputs include:
    - wave_front_*.csv: Detected wave fronts
    - wave_tail_*.csv: Detected wave tails
    - pair_data_*.csv: Paired wave observations
    - file_with_all_info_w_speed_*.csv: Complete pair data with trace IDs

    Args:
        file_base: Base filename identifier (e.g., "66477985b476f991aef3d7f0").
        lane_file: Lane file name (e.g., "WB_L2").
        file_root_local: Local root directory for outputs.
        file_path: Root path to input data.
        seg_speed: Critical segmentation speed in mph (default: 15).
        speed_tolerance: Speed tolerance threshold in mph for filtering peaks (default: 5.0).
    """
    print(f"{'='*50} Critical speed: {seg_speed} mph {'='*50}")

    # Load data - lane_file is now "WB_L{n}/vt_all.csv"
    csv_path = Path(file_path) / file_base / lane_file
    data = pd.read_csv(csv_path)
    
    # Extract lane number from path (e.g., "WB_L2/vt_all.csv" -> 2)
    lane_dir = Path(lane_file).parts[0] if "/" in lane_file or "\\" in lane_file else lane_file
    lane_number = int(lane_dir.split("_")[-1].replace("L", ""))
    
    print(f"Processing file: {file_base}, lane number: {lane_number}")

    # Setup output directory
    output_dir = Path(file_root_local) / "wave_cluster" / file_base / f"lane_{lane_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract wave points
    print("Step 1: Extracting wave fronts and tails...")
    wave_front, wave_tail = sgwcc_extract_wave_points(
        data, seg_speed=seg_speed, speed_tolerance=speed_tolerance
    )

    # Step 2: Save wave points
    print("Step 2: Saving wave points...")
    all_points = sgwcc_save_wave_points(wave_front, wave_tail, lane_number, seg_speed, output_dir)

    # Step 3: Pair wave observations
    print("Step 3: Pairing wave observations...")
    pair_data = sgwcc_pair_wave_points(all_points)
    pair_data_path = output_dir / f"pair_data_{seg_speed}.csv"
    pair_data.to_csv(pair_data_path, index=False)
    print(f"Saved: {pair_data_path}")

    # Step 4: Trace wave fronts
    print("Step 4: Tracing wave fronts...")
    wave_front_data = all_points[all_points["wave_type"] == 1].copy().reset_index(drop=True)
    wave_front_trace = sgwcc_trace_wave_trajectory(wave_front_data)
    wave_front_df = pd.DataFrame(
        wave_front_trace, columns=["time", "space", "vehicle_index", "trace_id", "unique_id"]
    )

    sgwcc_plot_trajectories(
        wave_front_df,
        output_dir / f"wave_front_traj_{seg_speed}.pdf",
        wave_type="front",
        seg_speed=seg_speed,
    )

    # Step 5: Trace wave tails
    print("Step 5: Tracing wave tails...")
    wave_tail_data = all_points[all_points["wave_type"] == 2].copy().reset_index(drop=True)
    wave_tail_trace = sgwcc_trace_wave_trajectory(wave_tail_data)
    wave_tail_df = pd.DataFrame(
        wave_tail_trace, columns=["time", "space", "vehicle_index", "trace_id", "unique_id"]
    )

    sgwcc_plot_trajectories(
        wave_tail_df,
        output_dir / f"wave_tail_traj_{seg_speed}.pdf",
        wave_type="tail",
        seg_speed=seg_speed,
    )

    # Step 6: Link traces to pair data
    print("Step 6: Linking traces to pair data...")
    pair_data = pd.merge(
        pair_data,
        wave_front_df[["trace_id", "unique_id"]],
        how="left",
        left_on="unique_index_front",
        right_on="unique_id",
    )
    pair_data = pair_data.drop(columns="unique_id").rename(columns={"trace_id": "trace_id_front"})

    pair_data = pd.merge(
        pair_data,
        wave_tail_df[["trace_id", "unique_id"]],
        how="left",
        left_on="unique_index_tail",
        right_on="unique_id",
    )
    pair_data = pair_data.drop(columns="unique_id").rename(columns={"trace_id": "trace_id_tail"})

    # Final output
    output_path = output_dir / f"file_with_all_info_w_speed_{seg_speed}.csv"
    pair_data.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"{'='*50} Processing complete {'='*50}")


def sgwcc_identify_connected_components(
    pair_data: pd.DataFrame, max_iterations: int = 300
) -> Tuple[List[int], List[int], List[List[int]]]:
    """Identify connected components of wave fronts and tails.

    Uses bidirectional graph traversal to find groups of fronts/tails that are
    connected through vehicle detections.

    Args:
        pair_data: DataFrame with trace_id_front and trace_id_tail columns.
        max_iterations: Maximum iterations for component propagation.

    Returns:
        Tuple of (front_traces, tail_traces, components) where each component
        is a list of indices in cc_result.
    """

    def get_unique_list(list_of_lists: List[List[int]]) -> List[int]:
        """Flatten and deduplicate a list of lists."""
        all_items = [item for sublist in list_of_lists for item in sublist]
        return list(set(all_items))

    all_front_trace = pair_data["trace_id_front"].unique().tolist()
    cc_result = []

    for front_trace_id in all_front_trace:
        if front_trace_id not in pair_data["trace_id_front"].values:
            continue

        temp = pair_data[pair_data["trace_id_front"] == front_trace_id].copy()
        front_trace = [[front_trace_id]]
        unique_tail_list = temp["trace_id_tail"].unique().tolist()

        for tail in unique_tail_list:
            temp_tail = pair_data[pair_data["trace_id_tail"] == tail]
            front_trace.append(temp_tail["trace_id_front"].unique().tolist())

        unique_front_list = get_unique_list(front_trace)
        tail_trace_update = [[x] for x in unique_tail_list]
        front_trace_update = [[x] for x in unique_front_list]

        # Iterative expansion to find all connected fronts/tails
        for _ in range(max_iterations):
            for front in unique_front_list:
                temp_front = pair_data[pair_data["trace_id_front"] == front]
                tail_trace_update.append(temp_front["trace_id_tail"].unique().tolist())

            unique_tail_list_update = get_unique_list(tail_trace_update)

            for tail in unique_tail_list_update:
                temp_tail = pair_data[pair_data["trace_id_tail"] == tail]
                front_trace_update.append(temp_tail["trace_id_front"].unique().tolist())

            unique_front_list_update = get_unique_list(front_trace_update)

            if set(unique_front_list_update) == set(unique_front_list):
                break

            unique_front_list = unique_front_list_update

        test_tail = pair_data[pair_data["trace_id_tail"].isin(unique_tail_list_update)].copy()
        test_front = pair_data[pair_data["trace_id_front"].isin(unique_front_list_update)].copy()

        if len(test_tail) >= 5 and len(test_front) >= 5:
            cc_result.append(
                {
                    "front_list": unique_front_list_update,
                    "tail_list": unique_tail_list_update,
                    "data": pd.concat([test_tail, test_front], axis=0),
                }
            )

            # Remove processed traces from pair_data
            pair_data = pair_data[~pair_data["trace_id_front"].isin(unique_front_list_update)]
            pair_data = pair_data[~pair_data["trace_id_tail"].isin(unique_tail_list_update)].reset_index(
                drop=True
            )

    return cc_result


def sgwcc_visualize_connected_components(
    pair_data: pd.DataFrame,
    vehicle_data: pd.DataFrame,
    cc_results: List[Dict],
    output_path: Path,
    seg_speed: int = 15,
) -> pd.DataFrame:
    """Visualize identified connected components on time-space diagram.

    Args:
        pair_data: Original pair data (for background trajectories).
        vehicle_data: Individual vehicle trajectories for context.
        cc_results: List of connected component results from sgwcc_identify_connected_components.
        output_path: Path to save visualization PDF.
        seg_speed: Critical speed (for labeling).

    Returns:
        Combined DataFrame with c_id column for all data in components.
    """
    # Avoid LaTeX dependency; use standard matplotlib text rendering
    plt.rc("text", usetex=False)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 60
    plt.figure(figsize=(45, 15))

    # Plot background trajectories (every 5th vehicle)
    for vehicle_id in vehicle_data["vehicle_index"].unique()[::5]:
        vdata = vehicle_data[vehicle_data["vehicle_index"] == vehicle_id].copy().reset_index(drop=True)
        plt.plot(vdata["time"], vdata["space"], color="k", alpha=0.4, lw=1)

    # Plot and label each connected component
    cc_combined = pd.DataFrame()
    for c_id, cc in enumerate(cc_results, 1):
        color = get_colors(c_id)
        component_data = cc["data"].copy()
        component_data["c_id"] = c_id
        cc_combined = pd.concat([cc_combined, component_data], axis=0)

        # Plot wave tails (circles)
        tail_data = component_data[component_data["wave_type_tail"] == 2]
        if len(tail_data) > 0:
            plt.scatter(
                tail_data["time_tail"],
                tail_data["space_tail"],
                color=color,
                marker="o",
                alpha=0.8,
                s=20,
            )

        # Plot wave fronts (triangles)
        front_data = component_data[component_data["wave_type_front"] == 1]
        if len(front_data) > 0:
            plt.scatter(
                front_data["time_front"],
                front_data["space_front"],
                color=color,
                marker="^",
                facecolors="none",
                alpha=0.5,
                s=50,
            )
            # Label component
            plt.text(
                front_data["time_front"].min() - 150,
                front_data["space_front"].min(),
                f"C{c_id}",
                fontsize=50,
            )

    plt.gca().invert_yaxis()
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Mile Marker")
    plt.grid(linestyle="--", lw=2, alpha=1)
    plt.xlim(0, 14400)
    plt.xticks(
        np.arange(0, 14401, 1800),
        ["6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00"],
    )
    plt.ylim(0, 4)
    plt.yticks(
        np.arange(0, 4.1, 0.5),
        ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"],
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return cc_combined


def sgwcc_stitch_components(
    file_root_local: str,
    file_base: str,
    lane_file: str,
    seg_speed: int,
    file_path: str,
) -> None:
    """Stitch connected wave components and visualize results.

    Args:
        file_root_local: Local root directory for outputs.
        file_base: Base filename identifier.
        lane_file: Lane file path (e.g., "WB_L2/vt_all.csv").
        seg_speed: Critical segmentation speed in mph.
        file_path: Root path to input vehicle trajectory data.
    """
    # Extract lane number from path (e.g., "WB_L2/vt_all.csv" -> 2)
    lane_dir = Path(lane_file).parts[0] if "/" in lane_file or "\\" in lane_file else lane_file
    lane_number = int(lane_dir.split("_")[-1].replace("L", ""))

    # Load pair data with traces
    pair_data_path = (
        Path(file_root_local)
        / "wave_cluster"
        / file_base
        / f"lane_{lane_number}"
        / f"file_with_all_info_w_speed_{seg_speed}.csv"
    )
    pair_data = pd.read_csv(pair_data_path)
    pair_data["trace_id_front"] = pair_data["trace_id_front"].astype("Int64")  # Nullable int
    pair_data["trace_id_tail"] = pair_data["trace_id_tail"].astype("Int64")
    pair_data = pair_data.dropna(subset=["trace_id_front", "trace_id_tail"])

    # Load vehicle data for background
    vehicle_data_path = Path(file_path) / file_base / lane_file
    vehicle_data = pd.read_csv(vehicle_data_path)

    # Identify connected components
    cc_results = sgwcc_identify_connected_components(pair_data)

    # Visualize components
    output_dir = Path(file_root_local) / "wave_cluster" / file_base / f"lane_{lane_number}"
    cc_vis_path = output_dir / f"analysis_{seg_speed}_lane_{lane_number}.pdf"
    cc_combined = sgwcc_visualize_connected_components(
        pair_data, vehicle_data, cc_results, cc_vis_path, seg_speed=seg_speed
    )

    # Save results
    cc_output_path = output_dir / f"CC_{seg_speed}.csv"
    cc_combined.to_csv(cc_output_path, index=False)
    print(f"Saved: {cc_output_path}")
