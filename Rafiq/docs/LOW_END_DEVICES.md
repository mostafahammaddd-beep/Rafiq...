# Low-end Android devices — status, proposed criteria, measurement protocol

**Nothing in this file has been measured on a phone.** The sandbox has no Android device, SDK, emulator or network access
to build the app. Items are labelled **VERIFIED** (read in the repo), **PROPOSED** (my recommendation, untested) or **BLOCKED**.

## What the phone does today (VERIFIED by reading the code)
* Captures 16 kHz mono PCM16 (`AudioRecord`), measures energy every 20 ms, drops silence after 500 ms (one `pause` frame
  instead), keeps 250 ms before speech and 500 ms after, sends ~160 ms frames over one OkHttp WebSocket.
* Runs **no neural inference**. RAM/CPU on the phone is therefore the UI (Jetpack Compose), the audio loop and the socket.
* Build config: `minSdk 24` (Android 7.0), `targetSdk 34`, `compileSdk 34`.
* No offline mode: after 3 failed reconnects (back-off 1 s, 2 s, 4 s; at most 10 retries in a whole session) it shows a connection error and gives no corrections (this is the correct
  behaviour when it cannot assess reliably — it never invents a correction).

## Proposed minimum device (PROPOSED — UNVERIFIED)
Android 7.0 (API 24, the build minimum) with ≥ 2 GB RAM and a working microphone, on a connection that sustains the uplink
below. Storage: APK size **not measured** (no build). These become a *supported* statement only after the protocol below passes.

Network (arithmetic, not measured): speech uplink ≤ 32 kB/s; a 65 s Ayat al-Kursi session ≤ ≈ 2.1 MB up, small JSON down.

## Adaptive inference policy
* **Implemented:** none beyond the above, because there is no on-device model to choose between.
* **PROPOSED, to be built only after a licence-cleared on-device model exists** (see `ARCHITECTURE_DECISION.md` §3):

| Condition (read at start-up: `ActivityManager.isLowRamDevice`, `memoryClass`, ABI, `Build.VERSION.SDK_INT`, benchmark of a 3 s dummy chunk) | Mode |
|---|---|
| Online, any device | cloud assessment (today's behaviour) |
| Offline, device passes the on-device benchmark (measured RTF p95 < 1 with ≥ 25 % headroom, peak RAM within budget) | on-device **first pass only**: tracking, skipped/extra words, madd timing from forced alignment; no sifat corrections |
| Offline, device fails the benchmark | explain the limitation; **no corrections** |
One coherent engine in all modes (same reference, alignment and abstention rules); only the acoustic backend differs.

## Measurement protocol (BLOCKED until hardware exists)
Device classes: **low** (Android 7–9, 2 GB, Cortex-A53 class), **mid** (Android 11–12, 4 GB), **high** (flagship, current Android).
≥ 3 physical devices per class where possible; record make/model, Android build, RAM, SoC, thermal state at start.

| Metric | How |
|---|---|
| Peak RAM, UI jank | `adb shell dumpsys meminfo com.rafiq.app` every 5 s over 10 sessions; `adb shell dumpsys gfxinfo` frame stats |
| CPU, thermal | Perfetto trace; `adb shell dumpsys thermalservice`; 10 back-to-back 65 s sessions to expose throttling |
| Battery | `adb shell dumpsys batterystats --reset` then 10 sessions; report mAh and screen-on baseline |
| Audio-loop health | count dropped `AudioRecord` reads and send-queue depth over 10 sessions |
| End-to-end latency | timestamp each 160 ms frame at capture and each `progress` message that first covers it; report p50/p95 per network (Wi-Fi, 4G) |
| On-device model (future) | model load time, peak RSS, RTF p95 over ≥ 20 sessions, accuracy deltas (fp32 vs INT8; chunk 8/16/24; CPU vs XNNPACK vs QNN) on the same human-reviewed set |

### Proposed release criteria per class (PROPOSED)
UI jank < 5 % slow frames; zero audio-loop drops in 10 sessions; p95 end-to-end word-tracking lag ≤ 3 words (the simulator's own
bound; real value unknown); no thermal throttling that changes behaviour across 10 consecutive sessions. A device class
**meets the criteria only when all rows are measured and pass.** Until then: **no device class is claimed to meet them.**
