#!/usr/bin/env python3
"""
Modal cost model for ONE 65-second live Ayat al-Kursi session (standalone and sharing a container).

What is MEASURED here:   the engine's redundancy factor R = seconds of audio the model processes per second of
                         speech, taken by running the real SessionEngine on the simulator at two cadences.
What is a PUBLISHED RATE: Modal CPU / memory prices (modal.com/pricing, checked 2026-10).
What is an ASSUMPTION:   core-seconds per audio-second (derived from the README's MEASURED fp32 RTF 0.352 on 4
                         threads), the INT8 speed-up, the speech fraction, the snapshot restore time and the
                         resident model size.  Every assumption is a command-line flag - change it and re-run.

Billing model used (stated, because the breakdown depends on it):
   CPU    = rate x max(reserved_cores x container_s , core_seconds_actually_used)
            (reservation part + burst part above the reservation)
   RAM    = rate x GiB x container_s            (the memory request)
   container_s = startup + session + scaledown_window
The script prints PASS / FAIL against the targets; it does not bend assumptions to reach them.
"""
import argparse, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CPU_RATE = 0.0000131        # $ per physical core-second
MEM_RATE = 0.00000222       # $ per GiB-second
GPU_T4_RATE = 0.000164      # $ per second (for the comparison row only)
RESERVED_CORES = 0.125      # cpu=(0.125, 2.0)
CORE_LIMIT = 2.0


def measure_redundancy(speech_s_target, rtf):
    """Run the real engine on the simulator; returns (window_seconds / speech_seconds, passes)."""
    import numpy as np
    import sim
    import modal_app as M
    ref = sim.build_ref()
    PAUSES = {6, 11, 18, 25, 31, 39, 43, 46}
    base = [ref.waqf[k] if (k in PAUSES and ref.waqf[k]) else ref.wasl[k] for k in range(ref.n)]
    probe = sim.Speech(ref, base, pauses=PAUSES, seed=3, pause_s=0.6)
    scale = max(0.6, speech_s_target / probe.total)
    sp0 = sim.Speech(ref, base, pauses=PAUSES, seed=3, pause_s=0.6, scale=scale)
    sp1, pf = sim.client_trim(sp0)                       # what the phone actually streams
    acc = [0.0, 0]

    def factory(r, rec, preset):
        def counted(w):
            acc[0] += len(w) / M.SR
            acc[1] += 1
            return rec(w)
        return M.SessionEngine(r, counted, preset)

    eng, _log, final, _ = sim.run_stream(ref, sp1, "strict", rtf=rtf, pauses=pf, engine_factory=factory)
    streamed = sp1.total
    return acc[0] / streamed, acc[1], streamed, final["summary"]["judged"]


def session_cost(session_s, startup_s, scaledown_s, mem_gib, used_core_s):
    """-> (container_s, cpu_reservation_$, cpu_burst_$, ram_$, total_$) for ONE container serving ONE session."""
    container_s = startup_s + session_s + scaledown_s
    reserved_core_s = RESERVED_CORES * container_s
    cpu_res = reserved_core_s * CPU_RATE
    cpu_burst = max(0.0, used_core_s - reserved_core_s) * CPU_RATE
    ram = mem_gib * container_s * MEM_RATE
    return container_s, cpu_res, cpu_burst, ram, cpu_res + cpu_burst + ram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-s", type=float, default=65.0)
    ap.add_argument("--speech-fraction", type=float, default=0.85, help="share of the session that is speech (rest = silence not streamed)")
    ap.add_argument("--startup-s", type=float, default=3.0, help="snapshot restore time that is billed")
    ap.add_argument("--scaledown-s", type=float, default=20.0, help="idle time billed after the last socket closes (modal_app.MODAL_SCALEDOWN_S)")
    ap.add_argument("--fp32-rtf-4thr", type=float, default=0.352, help="MEASURED (README): fp32 real-time factor on 4 threads")
    ap.add_argument("--threads-measured", type=int, default=4)
    ap.add_argument("--int8-speedup", type=float, default=1.8, help="ASSUMPTION: dynamic INT8 speed-up of the encoder")
    ap.add_argument("--resident-gib-int8", type=float, default=1.3, help="ASSUMPTION: resident set of the hybrid-INT8 model + runtime")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-measure", action="store_true", help="skip the simulator run (use --redundancy)")
    ap.add_argument("--redundancy", type=float, default=None)
    a = ap.parse_args()

    speech_s = a.session_s * a.speech_fraction
    core_s_per_audio_fp32 = a.fp32_rtf_4thr * a.threads_measured          # core-seconds per audio-second (upper bound)
    core_s_per_audio_int8 = core_s_per_audio_fp32 / a.int8_speedup

    print("=" * 100)
    print("RAFIQ v14.5 - Modal cost model for one %.0f s live session" % a.session_s)
    print("=" * 100)
    print("Published rates : CPU $%.7f /core-s   RAM $%.8f /GiB-s   (T4 GPU $%.6f /s, comparison only)" % (CPU_RATE, MEM_RATE, GPU_T4_RATE))
    print("Reservation     : cpu=(%.3f, %.1f) -> idle floor = $%.8f /s   (0.125 x $%.7f)" % (RESERVED_CORES, CORE_LIMIT, RESERVED_CORES * CPU_RATE, CPU_RATE))
    print("Session         : %.0f s wall, %.0f s of speech streamed, + %.1f s startup + %.1f s scaledown_window"
          % (a.session_s, speech_s, a.startup_s, a.scaledown_s))
    print("Compute (assumed): fp32 %.2f core-s per audio-s (README RTF %.3f x %d threads, measured)  |  INT8 %.2f (x%.1f, assumption)"
          % (core_s_per_audio_fp32, a.fp32_rtf_4thr, a.threads_measured, core_s_per_audio_int8, a.int8_speedup))

    # ------------------------------------------------------------------ redundancy measured on the real engine
    if a.redundancy is not None or a.no_measure:
        R_hop = R_cpu = a.redundancy if a.redundancy is not None else 1.0
        passes = 0
        print("Redundancy R    : %.2f (given)" % R_hop)
    else:
        rtf_cpu = core_s_per_audio_int8 / CORE_LIMIT                       # what a 2-core INT8 container can sustain
        R_hop, p1, streamed, j1 = measure_redundancy(speech_s, 0.02)       # unlimited CPU: a pass every 160 ms hop
        R_cpu, p2, _s, j2 = measure_redundancy(speech_s, rtf_cpu)          # CPU-bound cadence (2 cores, INT8)
        print("Redundancy R    : measured on the real SessionEngine (simulator, %.1f s streamed after silence gating)" % streamed)
        print("                  160 ms hop, unlimited CPU : R = %.2f  (%d passes, %d/50 words judged)" % (R_hop, p1, j1))
        print("                  CPU-bound cadence (rtf %.2f): R = %.2f  (%d passes, %d/50 words judged)" % (rtf_cpu, R_cpu, p2, j2))
        print("                  (v14.4 re-ran windows of up to 10 s; a single pass over every second would be R = 1.00)")
        import modal_app as M
        print("                  hop sweep, unlimited CPU (RAFIQ_HOP_S):", end="")
        keep = (M.HOP_S, M.HOP_SLOW_S)
        for hop in (0.16, 0.32, 0.64, 1.0):
            M.HOP_S, M.HOP_SLOW_S = hop, max(hop, 0.20)
            r, pp, _s2, jj = measure_redundancy(speech_s, 0.02)
            print("  %.2fs -> R=%.2f (%d/50)" % (hop, r, jj), end="")
        M.HOP_S, M.HOP_SLOW_S = keep
        print()
    print()

    def used(core_per_audio, R):
        return core_per_audio * R * speech_s

    cases = [
        ("INT8, 2 cores, CPU-bound cadence      ", used(core_s_per_audio_int8, R_cpu)),
        ("INT8, unlimited CPU, 160 ms hop       ", used(core_s_per_audio_int8, R_hop)),
        ("INT8, hypothetical R = 1.0 (one pass) ", used(core_s_per_audio_int8, 1.0)),
    ]
    print("-" * 100)
    print("A) COMPUTE per session (core-seconds the model needs)  [sets the CPU burst bill]")
    for name, cs in cases:
        wall_floor = cs / CORE_LIMIT
        print("   %s: %7.1f core-s  -> needs >= %.1f s of wall time on %.0f cores%s"
              % (name, cs, wall_floor, CORE_LIMIT, "   (MORE than the %.0f s session: cannot keep up in real time)" % a.session_s if wall_floor > a.session_s else ""))
    print()

    cs_main = cases[0][1]
    ram_options = [(0.25, "directive: 256 MiB"), (0.5, "directive: 512 MiB"), (0.75, "directive: 768 MiB"),
                   (1.5, "v14.5 default request (1536 MiB)"), (a.resident_gib_int8, "estimated INT8 resident set")]
    print("-" * 100)
    print("B) STANDALONE session cost = CPU reservation + CPU burst + RAM      (compute: %s)" % cases[0][0].strip())
    print("   %-34s %9s %12s %12s %12s %12s  %s" % ("RAM option", "container", "CPU reserv.", "CPU burst", "RAM", "TOTAL", "fits model?"))
    results = {}
    for gib, label in ram_options:
        c_s, res, burst, ram, tot = session_cost(a.session_s, a.startup_s, a.scaledown_s, gib, cs_main)
        fits = gib >= a.resident_gib_int8 * 0.98
        results[label] = tot
        print("   %-34s %7.0f s  $%.8f  $%.8f  $%.8f  $%.8f  %s" % (label, c_s, res, burst, ram, tot, "yes" if fits else "NO (OOM: model alone ~%.1f GiB)" % a.resident_gib_int8))
    c_s, res0, _b, ram0, _t = session_cost(a.session_s, a.startup_s, a.scaledown_s, 0.5, 0.0)
    print("   reservation-only floor (no compute, 512 MiB): $%.8f  <- the part of the target that the config alone can reach" % (res0 + ram0))
    print()

    print("-" * 100)
    print("C) CONCURRENT: %d sessions share one container (CPU reservation and RAM split; each session's compute is its own)" % a.concurrency)
    demand = a.concurrency * cs_main / a.session_s
    print("   aggregate CPU demand = %d x %.1f core-s / %.0f s = %.1f cores  vs  burst limit %.1f cores -> %s"
          % (a.concurrency, cs_main, a.session_s, demand, CORE_LIMIT,
             "SATURATED (sessions would lag)" if demand > CORE_LIMIT else "fits"))
    gib = 1.5
    c_s, res, burst, ram, _t = session_cost(a.session_s, a.startup_s, a.scaledown_s, gib, cs_main)
    per_res, per_ram = res / a.concurrency, ram / a.concurrency
    per_cpu = cs_main * CPU_RATE                      # own compute (replaces the burst, it is what the sessions actually burn)
    per_conc = per_res + per_ram + per_cpu
    print("   per session: reservation $%.8f + RAM $%.8f + own compute $%.8f = $%.8f" % (per_res, per_ram, per_cpu, per_conc))
    print()

    print("-" * 100)
    print("D) VERDICT against the targets")
    T1, T2a, T2b = 0.00025, 0.000015, 0.00002
    best_standalone = results["estimated INT8 resident set"]
    ok1 = best_standalone < T1
    ok2 = per_conc < T2a
    print("   standalone  < $%.5f : $%.6f  -> %s   (%.1fx %s the target)" % (T1, best_standalone, "PASS" if ok1 else "FAIL",
          max(best_standalone / T1, T1 / best_standalone), "over" if not ok1 else "under"))
    print("   concurrent  < $%.6f: $%.6f  -> %s   (and < $%.5f: %s)" % (T2a, per_conc, "PASS" if ok2 else "FAIL", T2b, "PASS" if per_conc < T2b else "FAIL"))
    floor_ok = (res0 + ram0) < T1
    print("   reservation-only floor < $%.5f : $%.6f -> %s  (this is all the container configuration controls)" % (T1, res0 + ram0, "PASS" if floor_ok else "FAIL"))
    min_cpu = cases[2][1] * CPU_RATE
    print("   hard lower bound: even ONE INT8 pass over %.0f s of speech needs %.1f core-s = $%.6f of CPU alone"
          % (speech_s, cases[2][1], min_cpu))
    print("   a T4 GPU billed for the same %.0f s would cost $%.5f (comparison, rate x container seconds)" % (a.session_s + a.startup_s + a.scaledown_s, GPU_T4_RATE * (a.session_s + a.startup_s + a.scaledown_s)))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
