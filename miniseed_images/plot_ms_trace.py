from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, read
from obspy.core.trace import Trace


def deduplicate_stream_by_id(stream: Stream) -> Stream:
    seen_ids: set[str] = set()
    unique_traces: list[Trace] = []

    for trace in stream:
        trace_id = str(trace.id)
        if trace_id in seen_ids:
            continue
        seen_ids.add(trace_id)
        unique_traces.append(trace)

    return Stream(traces=unique_traces)


def merge_stream_by_id(
    stream: Stream,
    *,
    fill_value: float | None = 0.0,
    method: int = 0,
) -> Stream:
    merged = stream.copy()
    merged.merge(method=method, fill_value=fill_value)
    return merged


def _time_axis_seconds_from_ref(trace: Trace, reference_start) -> np.ndarray:
    sample_count = int(trace.stats.npts)
    sampling_rate_hz = float(trace.stats.sampling_rate)

    offset_s = float(trace.stats.starttime - reference_start)
    return offset_s + (np.arange(sample_count, dtype=np.float64) / sampling_rate_hz)


def print_second_segment_length(original_stream: Stream, trace_id: str) -> None:
    segments = [tr for tr in original_stream if tr.id == trace_id]
    segments.sort(key=lambda tr: tr.stats.starttime)

    if len(segments) < 2:
        print(f"{trace_id}: no second segment (found {len(segments)} segment).")
        return

    second = segments[1]
    duration_s = float(second.stats.endtime - second.stats.starttime)

    print(
        f"{trace_id}: second segment npts={int(second.stats.npts)}, "
        f"duration={duration_s:.6f} s, "
        f"start={second.stats.starttime}, end={second.stats.endtime}"
    )


def plot_overlay_dedup_vs_merged(
    original_stream: Stream,
    *,
    trace_id: str,
    merge_fill_value: float | None = 0.0,
) -> None:
    """
    Overlay:
    - Deduplicated (first segment only) trace
    - Merged trace (with optional gap padding)

    Assumptions:
    - trace_id exists in original_stream
    - sampling rate is consistent within trace_id
    """
    # Print the second segment length (if it exists)
    print_second_segment_length(original_stream, trace_id)

    # Get deduped trace for this id
    deduped_stream = deduplicate_stream_by_id(original_stream)
    deduped_candidates = [tr for tr in deduped_stream if tr.id == trace_id]
    if not deduped_candidates:
        raise ValueError(f"trace_id {trace_id!r} not found in deduplicated stream.")
    deduped_trace = deduped_candidates[0]

    print(len(deduped_trace))

    # Get merged trace for this id
    merged_stream = merge_stream_by_id(original_stream, fill_value=merge_fill_value, method=0)
    merged_candidates = [tr for tr in merged_stream if tr.id == trace_id]
    if not merged_candidates:
        raise ValueError(f"trace_id {trace_id!r} not found in merged stream.")
    merged_trace = merged_candidates[0]

    print(len(merged_trace))

    # Reference start for a shared x-axis
    reference_start = min(deduped_trace.stats.starttime, merged_trace.stats.starttime)

    deduped_time_s = _time_axis_seconds_from_ref(deduped_trace, reference_start)
    merged_time_s = _time_axis_seconds_from_ref(merged_trace, reference_start)

    fig, axis = plt.subplots()
    axis.plot(deduped_time_s, deduped_trace.data, label="deduplicate (first segment)")
    axis.plot(merged_time_s, merged_trace.data, label=f"merge (fill_value={merge_fill_value})")

    axis.set_title(f"Overlay comparison: {trace_id}")
    axis.set_xlabel(f"Time since {reference_start} (s)")
    axis.set_ylabel("Amplitude")
    axis.legend()
    fig.tight_layout()
    plt.show()


def plot_with_second_segment(
    original_stream: Stream,
    *,
    trace_id: str,
    merge_fill_value: float | None = 0.0,
) -> None:
    # --- collect original segments ---
    segments = [tr for tr in original_stream if tr.id == trace_id]
    if not segments:
        raise ValueError(f"No traces found for {trace_id}")

    segments.sort(key=lambda tr: tr.stats.starttime)

    if len(segments) < 2:
        raise ValueError(f"{trace_id} has only one segment")

    first_segment = segments[0]
    second_segment = segments[1]

    # Print second segment length
    duration_s = float(second_segment.stats.endtime - second_segment.stats.starttime)
    print(
        f"{trace_id} second segment: "
        f"npts={int(second_segment.stats.npts)}, "
        f"duration={duration_s:.6f} s, "
        f"start={second_segment.stats.starttime}, "
        f"end={second_segment.stats.endtime}"
    )

    # --- deduplicated ---
    deduped_stream = deduplicate_stream_by_id(original_stream)
    deduped_trace = next(tr for tr in deduped_stream if tr.id == trace_id)

    # --- merged ---
    merged_stream = merge_stream_by_id(
        original_stream,
        fill_value=merge_fill_value,
        method=0,
    )
    merged_trace = next(tr for tr in merged_stream if tr.id == trace_id)

    # Shared reference for time axis
    reference_start = min(
        first_segment.stats.starttime,
        merged_trace.stats.starttime,
    )

    # Time axes
    deduped_time = _time_axis_seconds_from_ref(deduped_trace, reference_start)
    merged_time = _time_axis_seconds_from_ref(merged_trace, reference_start)
    second_time = _time_axis_seconds_from_ref(second_segment, reference_start)

    # --- plotting ---
    fig, (axis_top, axis_bottom) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(10, 6),
    )

    # Top: overlay
    axis_top.plot(
        deduped_time,
        deduped_trace.data,
        label="deduplicate (first trace file)",
    )
    axis_top.plot(
        merged_time,
        merged_trace.data,
        label=f"merge (fill_value={merge_fill_value})",
    )
    axis_top.set_title(f"{trace_id} – overlay")
    axis_top.set_ylabel("Amplitude")
    axis_top.legend()

    # Bottom: second segment only
    axis_bottom.plot(
        second_time,
        second_segment.data,
        label="second segment",
    )
    axis_bottom.set_title("Second segment only")
    axis_bottom.set_xlabel(f"Time since {reference_start} (s)")
    axis_bottom.set_ylabel("Amplitude")
    axis_bottom.legend()

    fig.tight_layout()
    plt.show()

###### Compute ##########    

stream = read("C:/Users/LIN241/Downloads/OCN_20240903T012000Z.ms")


# Plot deduplicate, merged and the leftover segment 
plot_with_second_segment(
    stream,
    trace_id="XX.G1.H1.FQE",
    merge_fill_value=None,
)

# Plot deduplicate and merged
plot_overlay_dedup_vs_merged(
    stream,
    trace_id="XX.G1.H1.FQZ",
    merge_fill_value=None,
)