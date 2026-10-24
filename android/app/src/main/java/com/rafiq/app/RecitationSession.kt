package com.rafiq.app

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.sqrt

/** نتيجة كلمة واحدة: status = ok | tashkeel | tajweed | letters | unchecked */
data class WordIssue(val type: String, val kind: String, val note: String, val focus: String)

data class WordResult(val index: Int, val status: String, val issues: List<WordIssue>)

/** عميل شبكة مشترك (يُعاد استخدامه لتفادي إنشاء خيوط جديدة في كل جلسة). */
object Net {
    val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(90, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
}

/**
 * جلسة تلاوة حيّة واحدة (WebSocket فقط):
 *  - تلتقط الصوت (16kHz / Mono / PCM16) وتقيس الطاقة كل 20ms، وترسل الصوت للخادم كل 160ms.
 *  - بوابة صمت «آمنة للتجويد»: ذاكرة حلقية 250ms قبل بدء الكلام + 500ms هامش بعد انتهائه، فلا تُقطع
 *    الحروف الخافتة ولا انحباس القلقلة ولا همزة القطع.
 *  - بعد أكثر من 500ms صمت متواصل يتوقف إرسال الصوت ويُرسل إطار تحكم واحد {"type":"pause",...}.
 *  - يحتفظ بكل ما أُرسل في الذاكرة؛ إن انقطع الاتصال أو أغلقه الخادم لخمولٍ طويل (paused_idle) أعاد الاتصال
 *    عند عودة الكلام وأرسل {"type":"resume","word_idx":k} ثم الصوت من آخر نقطة تثبيت فقط.
 *  - كل الاستدعاءات للـ Listener تتم على الخيط الرئيسي.
 */
class RecitationSession(
    private val appContext: Context,
    private val strictness: String,
    private val listener: Listener
) {
    interface Listener {
        /** 0 = اتصال، 1 = متصل، 2 = إعادة اتصال */
        fun onConnection(state: Int)
        fun onReady(words: List<String>)
        fun onProgress(tracked: Int, committed: Int, audioMs: Long, updates: List<WordResult>)
        fun onFinal(words: List<WordResult>, complete: Boolean, debug: String)
        fun onLevel(rms: Float, voiced: Boolean, audioMs: Long)
        fun onAutoStop()
        /** الخادم أغلق الاتصال لأن لا أحد يتكلم؛ سيُستأنف تلقائياً من الكلمة resumeWord عند عودة الكلام. */
        fun onPaused(resumeWord: Int)
        fun onError(message: String)
        fun onFatal(message: String)
    }

    /** عنصر واحد من صندوق الإرسال: إطار صوت PCM أو إطار تحكم pause (pcm == null). */
    private class Out(val pcm: ByteArray?, val pauseMs: Int, val audioEndMs: Long, val voicedEndMs: Long)

    private val main = Handler(Looper.getMainLooper())
    private val lock = Any()
    private val outbox = ArrayList<Out>()
    private var audioEndMs = 0L
    private var voicedEndMs = 0L
    private var sentIdx = 0
    private var stopSent = false

    // نقطة التثبيت (checkpoint) المرسلة من الخادم: كلمات مثبّتة + موضع الصوت (بالزمن المطلق للجلسة)
    @Volatile private var resumeWord = 0
    @Volatile private var resumeAudioAbsMs = 0L
    @Volatile private var serverBaseMs = 0L       // الزمن المطلق الذي بدأ عنده مخزن الصوت في الخادم الحالي
    private var resumeToSend = -1

    @Volatile private var socket: WebSocket? = null
    @Volatile private var open = false
    @Volatile private var capturing = false
    @Volatile private var stopRequested = false
    @Volatile private var captureDone = false
    @Volatile private var finished = false
    @Volatile private var captureStarted = false
    @Volatile private var idlePaused = false
    @Volatile private var attempts = 0
    private var record: AudioRecord? = null

    // ------------------------------------------------------------------ بدء / إيقاف

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        val sr = RecitationConfig.SAMPLE_RATE
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) return false
        val rec: AudioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sr,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(minBuf, sr * 2)
            )
        } catch (e: Exception) {
            return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return false
        }
        record = rec
        capturing = true
        connect()                     // الالتقاط يبدأ عند وصول «ready» من الخادم (انظر beginCapture)
        return true
    }

    /** يبدأ التقاط الصوت فقط حين يصبح الخادم جاهزاً، فلا يتراكم تسجيل طويل أثناء إيقاظه. */
    private fun beginCapture() {
        if (captureStarted || finished) return
        val rec = record ?: return
        captureStarted = true
        val t = Thread { captureLoop(rec) }
        t.name = "rafiq-capture"
        t.start()
    }

    private fun releaseUnstartedRecorder() {
        if (!captureStarted) {
            try {
                record?.release()
            } catch (_: Exception) {
            }
        }
    }

    /** المستخدم ضغط «إيقاف»: نكمل إرسال ما التُقط ثم نطلب التقرير النهائي. */
    fun stop() {
        stopRequested = true
        if (idlePaused && !finished) {
            // الخادم أغلق الاتصال وكل ما سُمع حُكم عليه سابقاً: لا حاجة لاتصال جديد
            finished = true
            capturing = false
            main.post { listener.onFinal(ArrayList<WordResult>(), false, "") }
        }
        pump()
    }

    /** إنهاء فوري دون انتظار التقرير. */
    fun abort() {
        finished = true
        capturing = false
        stopRequested = true
        try {
            socket?.cancel()
        } catch (_: Exception) {
        }
        socket = null
        open = false
        releaseUnstartedRecorder()
    }

    /** عدد ملّيثواني الصوت المنطوق (بعد حد الصمت) منذ لحظة audioMs (زمن مطلق من بداية التسجيل). */
    fun voicedSinceMs(audioMs: Long): Long {
        return synchronized(lock) {
            if (outbox.isEmpty()) {
                0L
            } else {
                val total = outbox[outbox.size - 1].voicedEndMs
                var lo = 0
                var hi = outbox.size - 1
                var at = -1
                while (lo <= hi) {
                    val mid = (lo + hi) ushr 1
                    if (outbox[mid].audioEndMs <= audioMs) {
                        at = mid
                        lo = mid + 1
                    } else {
                        hi = mid - 1
                    }
                }
                val atMark = if (at < 0) 0L else outbox[at].voicedEndMs
                total - atMark
            }
        }
    }

    // ------------------------------------------------------------------ التقاط الصوت

    /** بداية العنصر i بالملّيثانية (زمن مطلق). */
    private fun startMsOf(i: Int): Long = if (i <= 0) 0L else outbox[i - 1].audioEndMs

    /** أول عنصر ينبغي إعادة إرساله ليغطي الصوت من الموضع المطلق absMs. يُستدعى داخل lock. */
    private fun indexAtAudioMs(absMs: Long): Int {
        var i = outbox.size - 1
        while (i >= 0) {
            if (startMsOf(i) <= absMs) return i
            i--
        }
        return 0
    }

    /** يضيف ما تجمّع إلى صندوق الإرسال كإطار صوت واحد (<= 160ms). */
    private fun flushPending(pend: ByteArrayOutputStream, voicedMs: Long) {
        if (pend.size() == 0) return
        val bytes = pend.toByteArray()
        pend.reset()
        synchronized(lock) {
            audioEndMs += (bytes.size / 32).toLong()          // 16 عيّنة/ms × 2 بايت
            voicedEndMs += voicedMs
            outbox.add(Out(bytes, 0, audioEndMs, voicedEndMs))
        }
        if (voicedMs > 0L && idlePaused && !finished) {
            // عاد الكلام بعد أن أغلق الخادم الاتصال: أعد الاتصال الآن (وليس قبل ذلك فلا نُشغّل حاوية بلا داعٍ)
            idlePaused = false
            attempts = 0
            main.post { if (!finished) connect() }
        }
    }

    private fun addPauseFrame(durationMs: Int) {
        synchronized(lock) {
            outbox.add(Out(null, durationMs, audioEndMs, voicedEndMs))
        }
    }

    private fun captureLoop(rec: AudioRecord) {
        val sr = RecitationConfig.SAMPLE_RATE
        val sub = sr / 50                       // 20ms: دقة قرار الصمت/الكلام
        val hop = sub * HOP_SUB                 // 160ms: الإرسال للخادم
        val shorts = ShortArray(hop)
        var noise = 0.004f
        var elapsedMs = 0L
        var silent = 0
        var heardSpeech = false
        var announced = false
        val ring = ArrayDeque<ByteArray>()
        val pend = ByteArrayOutputStream()
        var pendVoicedMs = 0L
        try {
            rec.startRecording()
            while (capturing && !stopRequested && !finished) {
                var filled = 0
                while (filled < hop && capturing && !stopRequested) {
                    val r = rec.read(shorts, filled, hop - filled)
                    if (r < 0) {
                        fatal("تعذّر قراءة الميكروفون. تأكد أن تطبيقاً آخر لا يستخدمه.")
                        return
                    }
                    filled += r
                }
                if (filled < hop) break

                var hopSum = 0.0
                var hopVoiced = false
                for (k in 0 until HOP_SUB) {
                    val off = k * sub
                    var sum = 0.0
                    val bytes = ByteArray(sub * 2)
                    for (i in 0 until sub) {
                        val s = shorts[off + i].toInt()
                        sum += (s * s).toDouble()
                        bytes[2 * i] = (s and 0xFF).toByte()
                        bytes[2 * i + 1] = ((s shr 8) and 0xFF).toByte()
                    }
                    hopSum += sum
                    val rms = (sqrt(sum / sub) / 32768.0).toFloat()
                    if (rms < noise * 1.6f + 0.002f) noise = 0.95f * noise + 0.05f * rms
                    val thr = (noise * 3f).coerceIn(0.006f, 0.02f)
                    if (rms >= thr) {
                        // كلام: أرسل أولاً الذاكرة الحلقية (250ms قبل بدء الكلام) ثم الإطار الحالي
                        hopVoiced = true
                        while (ring.isNotEmpty()) pend.write(ring.removeFirst())
                        pend.write(bytes)
                        pendVoicedMs += SUB_MS
                        silent = 0
                        announced = false
                        heardSpeech = true
                    } else {
                        silent += 1
                        if (silent <= HANGOVER_SUB) {
                            pend.write(bytes)                 // هامش 500ms بعد الكلام (يُرسل كما هو)
                        } else {
                            if (heardSpeech && !announced) {
                                // أكثر من 500ms صمت متواصل بعد كلام: أوقف إرسال الصوت وأعلن وقفة حقيقية واحدة
                                flushPending(pend, pendVoicedMs)
                                pendVoicedMs = 0L
                                addPauseFrame((HANGOVER_SUB + 1) * SUB_MS)
                                announced = true
                            }
                            ring.addLast(bytes)
                            if (ring.size > PREROLL_SUB) ring.removeFirst()
                        }
                    }
                }
                flushPending(pend, pendVoicedMs)
                pendVoicedMs = 0L
                elapsedMs += (HOP_SUB * SUB_MS).toLong()
                val audioMs = synchronized(lock) { audioEndMs }
                pump()
                val shownMs = elapsedMs
                val hopRms = (sqrt(hopSum / hop) / 32768.0).toFloat()
                val voicedNow = hopVoiced
                main.post { if (!finished) listener.onLevel(hopRms, voicedNow, shownMs) }
                if (audioMs >= MAX_SESSION_MS && !stopRequested) {
                    stopRequested = true
                    main.post { if (!finished) listener.onAutoStop() }
                }
            }
        } catch (e: Exception) {
            fatal("تعذّر تشغيل الميكروفون.")
            return
        } finally {
            try {
                rec.stop()
            } catch (_: Exception) {
            }
            try {
                rec.release()
            } catch (_: Exception) {
            }
            captureDone = true
            pump()
        }
    }

    // ------------------------------------------------------------------ الشبكة

    private fun connect() {
        if (finished) return
        val url = RecitationConfig.webSocketUrl(appContext, strictness)
        val req = Request.Builder().url(url).build()
        val state = if (attempts == 0) 0 else 2
        main.post { if (!finished) listener.onConnection(state) }
        socket = Net.client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                synchronized(lock) {
                    open = true
                    stopSent = false
                    if (resumeWord > 0) {
                        // نقطة تثبيت عديمة الحالة: الخادم الجديد يبدأ من الكلمة resumeWord ويستلم الصوت منذ آخر تثبيت فقط
                        val i = indexAtAudioMs(resumeAudioAbsMs)
                        sentIdx = i
                        serverBaseMs = startMsOf(i)
                        resumeToSend = resumeWord
                    } else {
                        sentIdx = 0
                        serverBaseMs = 0L
                        resumeToSend = -1
                    }
                }
                main.post { if (!finished) listener.onConnection(1) }
                pump()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMessage(text)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                onSocketGone(webSocket)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onSocketGone(webSocket)
            }
        })
    }

    private fun onSocketGone(ws: WebSocket) {
        if (ws !== socket) return
        open = false
        if (finished || idlePaused) return          // الخمول: لا نعيد الاتصال حتى يعود الكلام
        attempts++
        if (attempts > 8) {
            fatal("تعذّر الاتصال بالخادم. تأكد من الإنترنت ثم أعد المحاولة.")
            return
        }
        main.post { if (!finished) listener.onConnection(2) }
        main.postDelayed({ if (!finished) connect() }, minOf(5000L, 700L * attempts))
    }

    /** يرسل إطار resume (إن لزم) ثم كل عناصر صندوق الإرسال بالترتيب، ثم رسالة الإيقاف إن طُلبت. */
    private fun pump() {
        synchronized(lock) {
            val ws = socket
            if (ws == null || !open) return
            if (resumeToSend >= 0) {
                if (!ws.send("{\"type\":\"resume\",\"word_idx\":" + resumeToSend + "}")) return
                resumeToSend = -1
            }
            while (sentIdx < outbox.size) {
                val o = outbox[sentIdx]
                val p = o.pcm
                val ok = if (p != null) {
                    ws.send(p.toByteString())
                } else {
                    ws.send(
                        "{\"type\":\"pause\",\"duration_ms\":" + o.pauseMs +
                            ",\"speech_tail_padding_ms\":" + HANGOVER_MS + "}"
                    )
                }
                if (!ok) return
                sentIdx++
            }
            if (stopRequested && captureDone && !stopSent) {
                if (ws.send("{\"type\":\"stop\"}")) stopSent = true
            }
        }
    }

    private fun handleMessage(text: String) {
        val o = try {
            JSONObject(text)
        } catch (e: Exception) {
            return
        }
        when (o.optString("type")) {
            "ready" -> {
                attempts = 0
                beginCapture()
                val arr = o.optJSONArray("words")
                val ws = ArrayList<String>()
                if (arr != null) for (i in 0 until arr.length()) ws.add(arr.optString(i, ""))
                main.post { if (!finished) listener.onReady(ws) }
            }
            "progress" -> {
                val tracked = o.optInt("tracked", -1)
                val committed = o.optInt("committed", 0)
                val base = serverBaseMs
                val audioMs = base + o.optLong("audio_ms", 0L)
                resumeWord = committed
                resumeAudioAbsMs = base + o.optLong("lock_ms", 0L)
                val updates = parseWords(o.optJSONArray("updates"))
                main.post { if (!finished) listener.onProgress(tracked, committed, audioMs, updates) }
            }
            "paused_idle" -> {
                // الخادم حكم على كل ما سمعه ثم أغلق الاتصال ليتوقف الحساب؛ نحفظ نقطة التثبيت ونستأنف عند عودة الكلام
                resumeWord = o.optInt("resume_word_idx", resumeWord)
                resumeAudioAbsMs = serverBaseMs + o.optLong("audio_ms", 0L)
                idlePaused = true
                val k = resumeWord
                main.post { if (!finished) listener.onPaused(k) }
            }
            "final" -> {
                finished = true
                capturing = false
                val words = parseWords(o.optJSONArray("words"))
                val summary = o.optJSONObject("summary")
                val complete = summary?.optBoolean("complete", false) ?: false
                val debugText = o.optJSONObject("debug")?.toString() ?: ""
                main.post { listener.onFinal(words, complete, debugText) }
                try {
                    socket?.close(1000, "done")
                } catch (_: Exception) {
                }
            }
            "error" -> {
                val msg = o.optString("message", "")
                main.post { if (!finished) listener.onError(msg) }
            }
        }
    }

    private fun parseWords(arr: JSONArray?): List<WordResult> {
        val out = ArrayList<WordResult>()
        if (arr == null) return out
        for (i in 0 until arr.length()) {
            val w = arr.optJSONObject(i) ?: continue
            val issues = ArrayList<WordIssue>()
            val ia = w.optJSONArray("issues")
            if (ia != null) {
                for (k in 0 until ia.length()) {
                    val io = ia.optJSONObject(k) ?: continue
                    issues.add(
                        WordIssue(
                            io.optString("t", "letters"),
                            io.optString("k", ""),
                            io.optString("n", ""),
                            io.optString("f", "")
                        )
                    )
                }
            }
            out.add(WordResult(w.optInt("i", -1), w.optString("s", "ok"), issues))
        }
        return out
    }

    private fun fatal(message: String) {
        finished = true
        capturing = false
        releaseUnstartedRecorder()
        main.post { listener.onFatal(message) }
    }

    companion object {
        private const val MAX_SESSION_MS = 300_000L   // 5 دقائق كحد أقصى
        private const val SUB_MS = 20                  // دقة قرار الصمت/الكلام
        private const val HOP_SUB = 8                  // 8 × 20ms = 160ms لكل إرسال
        private const val HANGOVER_MS = 500            // هامش بعد الكلام (ومدة الصمت قبل إعلان الوقفة)
        private const val HANGOVER_SUB = HANGOVER_MS / SUB_MS   // 25
        private const val PREROLL_SUB = 13             // ≈ 250ms قبل بدء الكلام
    }
}
