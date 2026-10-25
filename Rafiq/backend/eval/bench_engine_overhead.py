"""
Measures the cost of Rafiq's OWN engine code (alignment, judging, gate, bookkeeping) per audio second - everything except the
neural network. The neural model is replaced by the simulator's text-level fake recogniser, so this is NOT an end-to-end
latency and NOT a phone benchmark: it only bounds how much of the budget is spent outside the model on THIS machine.

    python3 eval/bench_engine_overhead.py [--sessions 10]
"""
import argparse, os, platform, resource, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dev"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import sim            # noqa: E402
import modal_app as M  # noqa: E402
import eval_stats as E  # noqa: E402


def cpu_name() -> str:
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    a = ap.parse_args()
    ref = sim.build_ref()
    pauses = {6, 11, 18, 25, 31, 39, 43, 46}
    base = [ref.waqf[k] if (k in pauses and ref.waqf[k]) else ref.wasl[k] for k in range(ref.n)]
    step_ms, proc_s, audio_s, n_steps = [], [], [], []
    for seed in range(a.sessions):
        sp = sim.Speech(ref, base, pauses=pauses, seed=seed)
        times = []

        def factory(ref_, rec, preset):
            eng = M.SessionEngine(ref_, rec, preset)
            orig = eng.step

            def timed(*args, **kw):
                t0 = time.perf_counter()
                try:
                    return orig(*args, **kw)
                finally:
                    times.append(time.perf_counter() - t0)
            eng.step = timed
            return eng

        t0 = time.perf_counter()
        eng, log, final, _ = sim.run_stream(ref, sp, engine_factory=factory)
        wall = time.perf_counter() - t0
        step_ms += [t * 1000 for t in times]
        proc_s.append(sum(times)); audio_s.append(sp.total); n_steps.append(len(times))
        assert final["summary"]["complete"], "simulated session did not complete"
    rep = E.rtf_report(proc_s, audio_s, "%s, 1 core, Python %s, numpy %s (sandbox; NOT a phone)" % (cpu_name(), platform.python_version(), __import__("numpy").__version__))
    sm = sorted(step_ms)
    print("sessions=%d  audio per session %.1f s (mean)  steps per session %.0f (mean)" % (a.sessions, statistics.mean(audio_s), statistics.mean(n_steps)))
    print("engine step() wall time, ms:  mean %.2f  p50 %.2f  p95 %.2f  max %.2f  (n=%d steps)" % (
        statistics.mean(sm), sm[len(sm) // 2], E.percentile_nearest_rank(sm, 95), sm[-1], len(sm)))
    print("engine-only RTF per session:  p50 %.4f  p95 %.4f  max %.4f   (n=%d sessions)" % (rep["p50"], rep["p95"], rep["max"], rep["n"]))
    print("peak RSS of this process: %.0f MiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    print("hardware: %s" % rep["hardware"])
    print("verdict helper: %s" % rep["verdict"])
    print("NOTE: the neural model is a text-level fake here; this bounds only the non-neural part of the budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
