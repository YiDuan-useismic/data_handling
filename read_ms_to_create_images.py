from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import os
import multiprocessing 
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FileTiming:
    path: Path
    starttime: object  # ObsPy UTCDateTime
    endtime: object    # ObsPy UTCDateTime



def _cast_stream_to_float32_inplace(stream: Stream) -> None:
    for trace in stream:
        trace.data = np.asarray(trace.data, dtype=np.float32)


def _validate_lowpass_cutoff(trace: Trace, *, cutoff_hz: float) -> None:
    sampling_rate_hz = float(trace.stats.sampling_rate)
    nyquist_hz = 0.5 * sampling_rate_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"lowpass cutoff_hz={cutoff_hz} must be < Nyquist={nyquist_hz:.1f} Hz "
            f"(sampling_rate={sampling_rate_hz:.1f} Hz) for trace {trace.id}."
        )


def lowpass_filter_stream_inplace(
    stream: Stream,
    *,
    cutoff_hz: float = 60.0,
    order: int = 2,
    zerophase: bool = True,
    margin_samples: int = 10,
) -> Stream:
    """
    Apply a Butterworth lowpass filter to each trace in-place and return the stream.

    Notes:
    - Mutates the input stream.
    - Returns the same stream for composability.
    """
    for trace in stream:
        _validate_lowpass_cutoff(trace, cutoff_hz=cutoff_hz)

        trace.filter(
            "lowpass",
            freq=cutoff_hz,
            corners=order,
            zerophase=zerophase,
        )

        if margin_samples > 0:
            trace.data = trace.data[margin_samples:]

        

    return stream


def align_stream_to_union(
    stream: Stream,
    *,
    fill_value: float = 0.0,
) -> Stream:
    if len(stream) == 0:
        raise ValueError("Stream is empty.")

    start_time = min(trace.stats.starttime for trace in stream)
    end_time = max(trace.stats.endtime for trace in stream)

    aligned = stream.copy()

    # Cast to float BEFORE padding so ObsPy creates float padding arrays.
    _cast_stream_to_float32_inplace(aligned)

    aligned.trim(
        starttime=start_time,
        endtime=end_time,
        pad=False,
        fill_value=float(fill_value),
    )

    aligned.traces.sort(
        key=lambda trace: (
            str(trace.stats.network),
            str(trace.stats.station),
            str(trace.stats.location),
            str(trace.stats.channel),
        )
    )


    return aligned


def _stream_to_matrix(stream: Stream) -> tuple[np.ndarray, list[str]]:
    min_samples = min(int(trace.stats.npts) for trace in stream)
    channel_labels: list[str] = []
    rows: list[np.ndarray] = []

    for trace in stream:
        channel_labels.append(str(trace.id))
        rows.append(np.asarray(trace.data[:min_samples], dtype=np.float32))

    matrix = np.stack(rows, axis=0)  # (channels, samples)
    return matrix, channel_labels


def _normalise_to_minus1_plus1(data: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(data)))
    if max_abs == 0.0:
        return data
    return data / max_abs


def _plot_window_as_image(
    window_matrix: np.ndarray,  # (channels, samples)
    trace_ids,
    output_path: Path,
    downsample_factor: int,
    dpi: int,
) -> None:
    channel_count = int(window_matrix.shape[0])
    figure_height = max(2.0, 0.6 * channel_count)

    figure, axes = plt.subplots(
        nrows=channel_count,
        ncols=1,
        sharex=True,
        figsize=(12, figure_height),
    )
    if channel_count == 1:
        axes = [axes]

    for axis, channel_row in zip(axes, window_matrix):
        data = channel_row
        if downsample_factor > 1:
            data = data[::downsample_factor]

        data = data - float(np.mean(data))
        data = _normalise_to_minus1_plus1(data)

        axis.plot(data, linewidth=0.6, color="black")


    # Hide absolutely everything, right before saving (robust against sharex quirks)
    for axis in axes:
        axis.set_axis_off()
        axis.set_frame_on(False)
        axis.patch.set_visible(False)
        for spine in axis.spines.values():
            spine.set_visible(False)


    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def deduplicate_stream_by_id(stream: Stream) -> Stream:
    """
    Keep the first occurrence of each trace.id (NET.STA.LOC.CHA),
    drop all subsequent duplicates.
    """
    seen_ids: set[str] = set()
    unique_traces: list[Trace] = []

    for trace in stream:
        trace_id = str(trace.id)
        if trace_id in seen_ids:
            continue  # drop duplicate
        seen_ids.add(trace_id)
        unique_traces.append(trace)

    return Stream(traces=unique_traces)

def export_overlapping_windows_across_files(
    input_file: Path,
    #output_file: Path,
    #pattern: str = "*.ms",
    window_seconds: float = 5.0,
    overlap_fraction: float = 0.5,
    fill_value: float = 0.0,
    downsample_factor: int = 1,
    dpi: int = 200,
) -> None:
    

    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0.0, 1.0).")

    # paths = sorted(input_dir.glob(pattern))
    # if not paths:
    #     raise ValueError(f"No files found in {input_dir} matching {pattern!r}")

    # Read just headers (still via read), extract timing for sorting
    #file_timings: list[FileTiming] = []
    # for path in paths:
    #stream = read(str(input_file))

    raw_stream = read(str(input_file))
    raw_stream = deduplicate_stream_by_id(raw_stream)
    aligned_stream = align_stream_to_union(raw_stream, fill_value=fill_value)
    print(len(aligned_stream))
    for tra in aligned_stream: 
        print(len(tra))

    stream = lowpass_filter_stream_inplace(aligned_stream, cutoff_hz=60.0, order=2, zerophase=False, margin_samples=20)

    start_time = min(trace.stats.starttime for trace in stream)
    end_time = max(trace.stats.endtime for trace in stream)
    file_timings = FileTiming(path=input_file, starttime=start_time, endtime=end_time)
    #file_timings.append(FileTiming(path=input_file, starttime=start_time, endtime=end_time))

    #file_timings.sort(key=lambda item: item.starttime)

    # Rolling buffer: (channels, buffered_samples)
    buffer_matrix: np.ndarray | None = None
    buffer_channel_labels: list[str] | None = None

    sampling_rate_hz: float | None = None
    samples_per_window: int | None = None
    samples_per_hop: int | None = None

    expected_next_start: object | None = None  # UTCDateTime
    window_index = 0

    # for item in file_timings:
    #stream = _align_stream_to_union(read(str(file_timings.path)), fill_value=fill_value)

    file_sampling_rate_hz = float(stream[0].stats.sampling_rate)
    if sampling_rate_hz is None:
        sampling_rate_hz = file_sampling_rate_hz
        samples_per_window = int(round(window_seconds * sampling_rate_hz))
        hop_seconds = window_seconds * (1.0 - overlap_fraction)
        samples_per_hop = int(round(hop_seconds * sampling_rate_hz))

        if samples_per_window <= 0 or samples_per_hop <= 0:
            raise ValueError("window_seconds/overlap_fraction produce invalid sample counts.")
    else:
        if abs(file_sampling_rate_hz - sampling_rate_hz) > 1e-6:
            raise ValueError(f"Sampling rate changed: {file_sampling_rate_hz} vs {sampling_rate_hz}")

    file_matrix, file_labels = _stream_to_matrix(stream)

    if buffer_matrix is None:
        buffer_matrix = file_matrix
        buffer_channel_labels = file_labels
        buffer_start_time = file_timings.starttime
        expected_next_start = file_timings.endtime + stream[0].stats.delta
    else:
        if file_labels != buffer_channel_labels:
            raise ValueError(
                "Channel set/order changed between files.\n"
                f"Previous: {buffer_channel_labels}\n"
                f"Current:  {file_labels}"
            )

        # Handle gaps/overlaps in time between files (based on header times)
        dt_seconds = float(file_timings.starttime - expected_next_start)
        sample_offset = int(round(dt_seconds * sampling_rate_hz))

        if sample_offset > 0:
            # Gap: pad with fill_value so continuity holds
            gap = np.full((buffer_matrix.shape[0], sample_offset), fill_value, dtype=np.float32)
            buffer_matrix = np.concatenate([buffer_matrix, gap, file_matrix], axis=1)
        elif sample_offset < 0:
            # Overlap: drop the first -sample_offset samples from the new file
            drop = min(-sample_offset, file_matrix.shape[1])
            file_matrix = file_matrix[:, drop:]
            buffer_matrix = np.concatenate([buffer_matrix, file_matrix], axis=1)
        else:
            buffer_matrix = np.concatenate([buffer_matrix, file_matrix], axis=1)

        expected_next_start = file_timings.endtime

    while buffer_matrix.shape[1] >= samples_per_window:
        window = buffer_matrix[:, :samples_per_window]


        # Absolute start time of this window
        window_start_time = buffer_start_time

        timestamp_str = window_start_time.strftime("%Y%m%dT%H%M%S")
        fractional = int(window_start_time.microsecond / 1000)  # ms

        #print (input_file)
        output_dir = Path(input_file).parents[0] / "image"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        out_path = output_dir/ f"window_{timestamp_str}_{fractional:03d}.jpg"
        #out_path = output_dir / f"window_{timestamp_str}_{fractional:03d}.jpg"

        _plot_window_as_image(
            window,
            trace_ids=buffer_channel_labels,
            output_path=out_path,
            downsample_factor=downsample_factor,
            dpi=dpi,
        )

        window_index += 1

        # Advance buffer and its start time by the hop
        buffer_matrix = buffer_matrix[:, samples_per_hop:]
        buffer_start_time = buffer_start_time + (samples_per_hop / sampling_rate_hz)

        print(f"Saved {window_index} windows to {output_dir}")



def get_files_to_process(data_dir, extension=".ms"):
    """
    Get all files with the specified extension from the source directory for processing.
    """
    files_to_process = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(extension):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    return files_to_process


def plot_and_save_multiprocess(
    data_dir, num_processes=None, file_extension=".ms"
):
    """
    Use multiple processes to read data, plot, and save figures as .jpg files.
    """
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)

    files_to_process = get_files_to_process(
        data_dir, extension=file_extension
    )

    # Use multiprocessing to parallelize the plotting and saving
    with multiprocessing.Pool(
        processes=num_processes or multiprocessing.cpu_count()
    ) as pool:
        pool.map(export_overlapping_windows_across_files, files_to_process)


def run_for_hours(start_dt, end_dt, dry_run=True):
    current = start_dt
    while current <= end_dt:
        year  = current.year
        month = f"{current.month:02d}"
        day   = f"{current.day:02d}"
        hour  = f"{current.hour:02d}"

        data_directory = Path(
            f"W:/work/data_archive/geophone_combined/OCN/"
            f"{year}/{month}/{day}/{hour}"
        )
        # output_directory = (
        #     f"W:/work/data_archive/geophone_combined/OCN/{month}/{day}/{hour}"
        # )

        if dry_run:
            # Just print what would be done
            print(f"{current} →\n"
                  f"  Data dir:   {data_directory}\n")
                  #f"  Output dir: {output_directory}\n")
        else:
            # Actual processing
            plot_and_save_multiprocess(
                data_directory,
                #output_directory,
                num_processes=12,
                file_extension=".ms",
            )

        current += timedelta(hours=1)

if __name__ == "__main__":
    # start_dt = datetime(2024, 9, 3, 10)  
    # end_dt   = datetime(2024, 9, 3, 23)  

    # # Pass dry_run=True to only print
    # run_for_hours(start_dt, end_dt, dry_run=False)


    export_overlapping_windows_across_files(Path("C:/Users/LIN241/Downloads/OCN_20240903T012000Z.ms"))