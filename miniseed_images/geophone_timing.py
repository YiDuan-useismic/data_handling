from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from obspy import UTCDateTime, read
from obspy.core.stream import Stream
from obspy.core.trace import Trace


@dataclass(frozen=True)
class TraceTimingSummary:
    trace_id: str
    start: UTCDateTime
    end: UTCDateTime
    npts: int
    sampling_rate_hz: float
    delta_s: float
    computed_end: UTCDateTime
    end_mismatch_s: float


def _trace_id(trace: Trace) -> str:
    return f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"


def summarise_trace_timing(trace: Trace) -> TraceTimingSummary:
    start = trace.stats.starttime
    end = trace.stats.endtime
    npts = int(trace.stats.npts)
    sampling_rate_hz = float(trace.stats.sampling_rate)
    delta_s = float(trace.stats.delta)

    # ObsPy defines endtime as time of last sample.
    computed_end = start + (npts - 1) * delta_s if npts > 0 else start
    end_mismatch_s = float(end - computed_end)

    return TraceTimingSummary(
        trace_id=_trace_id(trace),
        start=start,
        end=end,
        npts=npts,
        sampling_rate_hz=sampling_rate_hz,
        delta_s=delta_s,
        computed_end=computed_end,
        end_mismatch_s=end_mismatch_s,
    )


def print_stream_timing_report(stream: Stream) -> None:
    summaries = [summarise_trace_timing(tr) for tr in stream]

    # Sort by id then start
    summaries.sort(key=lambda item: (item.trace_id, item.start.timestamp))

    print("Trace timing summary:")
    for item in summaries:
        print(
            f"- {item.trace_id} | start={item.start} | end={item.end} | "
            f"npts={item.npts} | fs={item.sampling_rate_hz:.6f} Hz | "
            f"end_mismatch={item.end_mismatch_s:+.6e} s"
        )

    # Identify the union window
    union_start = min(item.start for item in summaries) if summaries else None
    union_end = max(item.end for item in summaries) if summaries else None
    print()
    print(f"Union window: start={union_start} end={union_end}")

    # Group by trace id (channel) and report total duration and gaps/overlaps
    print()
    print("Per-channel gaps/overlaps (using ObsPy get_gaps):")
    gaps = stream.get_gaps()  # list of tuples
    if not gaps:
        print("- None reported by get_gaps()")
    else:
        for gap in gaps:
            # Format per ObsPy docs: (net, sta, loc, cha, t1, t2, delta, samples)
            net, sta, loc, cha, t1, t2, delta, samples = gap
            gap_id = f"{net}.{sta}.{loc}.{cha}"
            kind = "GAP" if samples > 0 else "OVERLAP"
            print(
                f"- {kind} {gap_id} | {t1} -> {t2} | "
                f"delta={delta:.6f} s | samples={samples}"
            )


def check_alignment_to_sample_grid(
    stream: Stream,
    *,
    reference_start: UTCDateTime | None = None,
    tolerance_fraction_of_sample: float = 0.25,
) -> None:
    """
    Checks whether each trace start time lands on the same sample grid.

    reference_start: if None, uses the earliest start time in the stream.
    tolerance_fraction_of_sample: e.g. 0.25 means within 1/4 sample is "aligned".
    """
    if len(stream) == 0:
        print("Stream is empty.")
        return

    start_times = [tr.stats.starttime for tr in stream]
    ref_start = reference_start or min(start_times)

    print("Sample-grid alignment check:")
    for trace in stream:
        delta_s = float(trace.stats.delta)
        offset_s = float(trace.stats.starttime - ref_start)

        # Distance to nearest sample tick
        nearest_ticks = round(offset_s / delta_s) if delta_s > 0 else 0
        nearest_s = nearest_ticks * delta_s
        residual_s = offset_s - nearest_s

        is_aligned = abs(residual_s) <= tolerance_fraction_of_sample * delta_s
        status = "OK" if is_aligned else "MISALIGNED"

        print(
            f"- {_trace_id(trace)} | offset={offset_s:+.6f} s | "
            f"residual={residual_s:+.6e} s | {status}"
        )


def main(miniseed_path: str) -> None:
    stream = read(miniseed_path)
    print_stream_timing_report(stream)
    print()
    check_alignment_to_sample_grid(stream)


# Example:
main("C:/Users/LIN241/Downloads/OCN_20240903T012000Z.ms")
