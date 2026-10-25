package com.rafiq.app

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.math.floor
import kotlin.math.max

data class Summary(
    val ok: Int,
    val tashkeel: Int,
    val tajweed: Int,
    val letters: Int,
    val unchecked: Int,
    val judged: Int,
    val total: Int
)

enum class Stage { IDLE, STARTING, LISTENING, FINISHING, DONE }

/**
 * يجمع حالة الشاشة كلها. الفكرة:
 *  - الخادم يؤكد نتيجة كل كلمة مرة واحدة فقط (أخضر/أصفر/برتقالي/أحمر).
 *  - التوهج الذهبي (مكان القارئ الآن) يتقدم بسلاسة بين تحديثات الخادم اعتماداً على
 *    سرعة القارئ ومدة الصوت المنطوق فعلاً؛ فلا يظهر أي تأخر شبكة للمستخدم.
 */
class RafiqController(private val appContext: Context) : RecitationSession.Listener {

    private val main = Handler(Looper.getMainLooper())

    var stage by mutableStateOf(Stage.IDLE)
        private set
    var serverState by mutableStateOf(0)          // 0 تجهيز، 1 جاهز، 2 غير متاح، 3 نائم (وضع التوفير)
        private set
    var connection by mutableStateOf(0)           // 0 اتصال، 1 متصل، 2 إعادة اتصال
        private set
    var message by mutableStateOf("")
        private set
    var words by mutableStateOf(WORDS)
        private set
    var results by mutableStateOf<List<WordResult?>>(List(WORDS.size) { null })
        private set
    var glow by mutableStateOf(-1)
        private set
    var level by mutableStateOf(0f)
        private set
    var seconds by mutableStateOf(0)
        private set
    var playing by mutableStateOf(false)
        private set
    var strictness by mutableStateOf(RecitationConfig.getStrictness(appContext))
        private set

    private var session: RecitationSession? = null
    private var player: MediaPlayer? = null
    private var released = false
    private var serverTracked = -1
    private var serverCommitted = 0
    private var serverAudioMs = 0L
    private var pace = 1.3                  // كلمة/ثانية (تُقدَّر أثناء التلاوة)
    private var silentMs = 0
    private var smoothedLevel = 0f
    private var lastDebug = ""
    private val reports = ArrayList<String>()
    private val finalTimeout = Runnable { onFinalTimeout() }

    // ------------------------------------------------------------------ تجهيز الخادم

    /** وضع التوفير: الخادم المدفوع بالثانية يبقى نائماً حتى يطلب المستخدم إيقاظه. */
    fun markSleeping() {
        serverState = 3
    }

    private fun afterSession() {
        if (RecitationConfig.isMetered(appContext)) serverState = 3
    }

    // ------------------------------------------------------------------ التلاوة

    fun startRecitation() {
        if (stage == Stage.STARTING || stage == Stage.LISTENING || stage == Stage.FINISHING) return
        stopReference()
        resetRun()
        val s = RecitationSession(appContext, strictness, this)
        if (!s.start()) {                         // المصافحة المبكرة مع الخادم تبدأ هنا (لحظة لمس الزر)
            message = "تعذّر تشغيل الميكروفون. تأكد من منح الإذن."
            return
        }
        session = s
        val warm = serverState == 1
        stage = Stage.STARTING
        glow = -1
        if (serverState == 3) serverState = 0
        message = if (warm) "جارٍ الاتصال بالخادم…" else "الخادم يستيقظ… انتظر لحظات (حتى نصف دقيقة) ثم ابدأ التلاوة"
    }

    fun stopRecitation() {
        val s = session ?: return
        if (stage != Stage.LISTENING) return
        stage = Stage.FINISHING
        message = "جارٍ إعداد النتيجة…"
        level = 0f
        s.stop()
        main.removeCallbacks(finalTimeout)
        main.postDelayed(finalTimeout, 220000L)
    }

    /** المستخدم ضغط الميكروفون أثناء انتظار الخادم: نلغي البدء. */
    fun cancelStart() {
        if (stage != Stage.STARTING) return
        session?.abort()
        session = null
        stage = Stage.IDLE
        message = ""
        afterSession()
    }

    fun cancelIfActive() {
        stopReference()
        if (stage == Stage.STARTING) cancelStart() else if (stage == Stage.LISTENING) stopRecitation()
    }

    fun release() {
        released = true
        main.removeCallbacks(finalTimeout)
        session?.abort()
        session = null
        stopReference()
    }

    private fun resetRun() {
        results = List(words.size) { null }
        glow = -1
        level = 0f
        seconds = 0
        serverTracked = -1
        serverCommitted = 0
        serverAudioMs = 0L
        pace = 1.3
        silentMs = 0
        smoothedLevel = 0f
        message = ""
    }

    private fun onFinalTimeout() {
        if (stage != Stage.FINISHING) return
        session?.abort()
        session = null
        stage = Stage.DONE
        glow = -1
        afterSession()
        message = "تأخّر الخادم في إرسال التقرير الكامل؛ هذه النتائج التي وصلت حتى الآن."
    }

    // ------------------------------------------------------------------ RecitationSession.Listener

    override fun onConnection(state: Int) {
        connection = state
        if (stage == Stage.STARTING && state == 2) message = "إعادة محاولة الاتصال بالخادم…"
        if (stage == Stage.LISTENING) {
            if (state == 2) message = "انقطع الاتصال… نعيد المحاولة تلقائياً (لا تتوقف عن التلاوة)"
            else if (state == 1 && (message.startsWith("انقطع") || message.startsWith("الخادم في وضع"))) message = "تفضّل بالتلاوة…"
        }
    }

    override fun onReady(words: List<String>) {
        serverState = 1
        if (words.isNotEmpty() && words.size != WORDS.size) {
            // احتياط: لو اختلف تقسيم الكلمات بين الخادم والتطبيق نعرض كلمات الخادم كما هي
            this.words = words
            results = List(words.size) { null }
        }
        if (stage == Stage.STARTING) {            // الخادم جاهز: يبدأ الالتقاط الآن
            stage = Stage.LISTENING
            glow = 0
            message = "تفضّل بالتلاوة…"
        }
    }

    override fun onProgress(tracked: Int, committed: Int, audioMs: Long, updates: List<WordResult>) {
        if (stage != Stage.LISTENING && stage != Stage.FINISHING) return
        applyUpdates(updates)
        // قدّر سرعة القارئ (كلمات لكل ثانية نطق فعلي) من تقدم الخادم
        if (tracked > serverTracked && serverTracked >= 0) {
            val s = session
            if (s != null) {
                val voicedBetween = s.voicedSinceMs(serverAudioMs) - s.voicedSinceMs(audioMs)
                if (voicedBetween > 600L) {
                    val sample = (tracked - serverTracked) * 1000.0 / voicedBetween
                    pace = 0.6 * pace + 0.4 * sample.coerceIn(0.6, 3.2)
                }
            }
        }
        if (tracked > serverTracked) serverTracked = tracked
        if (committed > serverCommitted) serverCommitted = committed
        if (audioMs > serverAudioMs) serverAudioMs = audioMs
        updateGlow()
    }

    override fun onLevel(rms: Float, voiced: Boolean, audioMs: Long) {
        smoothedLevel = 0.6f * smoothedLevel + 0.4f * rms
        level = smoothedLevel
        seconds = (audioMs / 1000L).toInt()
        if (voiced) silentMs = 0 else silentMs += 160           // كل استدعاء = 160ms
        updateGlow()
    }

    override fun onAutoStop() {
        stopRecitation()
    }

    override fun onFinal(words: List<WordResult>, complete: Boolean, debug: String) {
        main.removeCallbacks(finalTimeout)
        if (debug.isNotEmpty()) lastDebug = debug
        afterSession()
        applyUpdates(words)
        session = null
        stage = Stage.DONE
        level = 0f
        glow = -1
        val sm = summary()
        message = if (sm.judged == 0) "لم أسمع تلاوة واضحة. اقترب من الميكروفون وأعد المحاولة." else ""
    }

    override fun onPaused(resumeWord: Int) {
        // الخادم أغلق الاتصال لخمول أكثر من 8 ثوانٍ (لا فواتير على هاتف متروك)؛ تُستأنف الجلسة تلقائياً من الكلمة resumeWord
        if (stage == Stage.LISTENING) {
            message = "الخادم في وضع الانتظار لتوفير الموارد — تابع التلاوة من الكلمة " + (resumeWord + 1) + " وسيستأنف تلقائياً"
        }
    }

    override fun onError(message: String) {
        // أخطاء مؤقتة في معالجة مقطع واحد؛ الجلسة تستمر
        if (stage == Stage.LISTENING) this.message = "تعذّر تحليل مقطع، سنواصل…"
    }

    override fun onFatal(message: String) {
        main.removeCallbacks(finalTimeout)
        session = null
        stage = if (summary().judged > 0) Stage.DONE else Stage.IDLE
        level = 0f
        glow = -1
        afterSession()
        this.message = message
    }

    // ------------------------------------------------------------------ مساعدات

    private fun applyUpdates(list: List<WordResult>) {
        if (list.isEmpty()) return
        val cur = results.toMutableList()
        for (u in list) if (u.index in cur.indices) cur[u.index] = u
        results = cur
    }

    private fun updateGlow() {
        if (stage != Stage.LISTENING) return
        val last = words.size - 1
        val base = max(0, max(serverTracked, serverCommitted)).coerceAtMost(last)
        val s = session
        val voicedAfter = if (s != null) s.voicedSinceMs(serverAudioMs) else 0L
        val extra = floor(pace * voicedAfter / 1000.0).toInt().coerceIn(0, 2)
        val cand = (base + extra).coerceAtMost(last)
        glow = if (cand >= glow) cand else if (silentMs >= 700) cand else glow
    }

    fun summary(): Summary {
        var ok = 0
        var ta = 0
        var tj = 0
        var le = 0
        var un = 0
        for (r in results) {
            if (r == null) continue
            when (r.status) {
                "ok" -> ok++
                "tashkeel" -> ta++
                "tajweed" -> tj++
                "letters" -> le++
                else -> un++
            }
        }
        return Summary(ok, ta, tj, le, un, ok + ta + tj + le + un, words.size)
    }

    fun changeStrictness(value: String) {
        RecitationConfig.saveStrictness(appContext, value)
        strictness = value
    }

    fun saveAccessToken(token: String) {
        RecitationConfig.saveAccessToken(appContext, token)
    }

    fun saveServerUrl(url: String) {
        RecitationConfig.saveServerBaseUrl(appContext, url)
        serverState = 3           // نائم: لا نتصل بالخادم الجديد إلا لحظة الضغط على التسجيل (لا استدعاءات تسخين)
    }

    /** يمسح حالة «لا اتصال» فقط. لا شبكة هنا: الاتصال يبدأ لحظة الضغط على زر التسجيل وحدها. */
    fun resetServerState() {
        serverState = 3
        message = ""
    }

    /** المستخدم يخبرنا أن حكم الخادم على كلمة كان خاطئاً: kind = missed | wrong_flag */
    fun reportWord(index: Int, kind: String) {
        val r = results.getOrNull(index)
        val o = JSONObject()
        o.put("word_index", index)
        o.put("word", words.getOrElse(index) { "" })
        o.put("shown_status", r?.status ?: "")
        o.put("user_says", if (kind == "missed") "has_an_undetected_mistake" else "was_recited_correctly")
        reports.add(o.toString())
        message = "سُجّلت ملاحظتك على الكلمة ✓ (ستُضاف للتقرير الفني)"
    }

    // ------------------------------------------------------------------ استماع التلاوة المرجعية

    fun toggleReference() {
        if (playing) {
            stopReference()
            return
        }
        if (stage == Stage.LISTENING || stage == Stage.FINISHING) return
        val mp = MediaPlayer()
        try {
            mp.setAudioAttributes(
                AudioAttributes.Builder()
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .build()
            )
            mp.setDataSource(AUDIO_URL)
            mp.setOnPreparedListener { p ->
                p.start()
                playing = true
                message = ""
            }
            mp.setOnCompletionListener { stopReference() }
            mp.setOnErrorListener { _, _, _ ->
                stopReference()
                message = "تعذّر تشغيل التلاوة المرجعية. تأكد من الإنترنت."
                true
            }
            player = mp
            message = "جارٍ تحميل التلاوة…"
            mp.prepareAsync()
        } catch (e: Exception) {
            try {
                mp.release()
            } catch (_: Exception) {
            }
            player = null
            message = "تعذّر تشغيل التلاوة المرجعية."
        }
    }

    fun stopReference() {
        val p = player
        player = null
        if (p != null) {
            try {
                p.stop()
            } catch (_: Exception) {
            }
            try {
                p.release()
            } catch (_: Exception) {
            }
        }
        playing = false
    }

    // ------------------------------------------------------------------ تقرير فني (للتشخيص)

    fun copyDebugReport() {
        val userReports = "[" + reports.joinToString(",") + "]"
        val stored = lastDebug
        if (stored.isNotEmpty()) {
            copyToClipboard("{\"user_reports\":" + userReports + ",\"session\":" + stored + "}")
            message = "تم نسخ التقرير الفني. الصقه في المحادثة."
            return
        }
        message = "جارٍ جلب التقرير الفني…"
        val url = RecitationConfig.debugUrl(appContext)
        Thread {
            var text: String? = null
            try {
                val client = Net.client.newBuilder().readTimeout(60, TimeUnit.SECONDS).build()
                client.newCall(Request.Builder().url(url).build()).execute().use { r ->
                    if (r.isSuccessful) text = r.body?.string()
                }
            } catch (e: Exception) {
                text = null
            }
            val result = text
            main.post {
                if (result == null) {
                    message = "تعذّر جلب التقرير الفني. أكمل تلاوة أولاً ثم أعد المحاولة."
                } else {
                    copyToClipboard("{\"user_reports\":" + userReports + ",\"session\":" + result + "}")
                    message = "تم نسخ التقرير الفني. الصقه في المحادثة."
                }
            }
        }.start()
    }

    private fun copyToClipboard(text: String) {
        val cm = appContext.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("rafiq-report", text))
    }
}
