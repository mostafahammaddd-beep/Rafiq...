package com.rafiq.app

import android.content.Context

/**
 * إعدادات الاتصال بالخادم (Modal) - تلاوة حيّة عبر WebSocket فقط. لا يحتاج المستخدم لأي مفاتيح أو إعدادات.
 * يمكن تغيير الرابط من ⚙ في أعلى التطبيق عند الحاجة فقط.
 */
object RecitationConfig {
    const val BACKEND_BASE_URL: String =
        "https://mostafahammad480--rafiq-quran-muaalem-backend-fastapi-asgi-app.us-east.modal.run"
    const val SURA_IDX: Int = 2
    const val AYA_IDX: Int = 255
    const val SAMPLE_RATE: Int = 16000

    private const val PREFS_NAME = "rafiq_prefs_v14"
    private const val KEY_URL = "server_url"
    private const val KEY_STRICT = "strictness"
    private const val KEY_TOKEN = "access_token"

    fun getServerBaseUrl(ctx: Context): String {
        val prefs = ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val saved = prefs.getString(KEY_URL, "")?.trim() ?: ""
        return normalizeBaseUrl(if (saved.isNotEmpty()) saved else BACKEND_BASE_URL)
    }

    fun saveServerBaseUrl(ctx: Context, rawUrl: String) {
        val normalized = normalizeBaseUrl(if (rawUrl.isBlank()) BACKEND_BASE_URL else rawUrl)
        ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_URL, normalized)
            .apply()
    }

    /** relaxed | balanced | strict */
    fun getStrictness(ctx: Context): String {
        val prefs = ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val s = prefs.getString(KEY_STRICT, "balanced") ?: "balanced"
        return if (s == "relaxed" || s == "strict") s else "balanced"
    }

    fun saveStrictness(ctx: Context, value: String) {
        ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_STRICT, value)
            .apply()
    }

    /**
     * رمز الدخول (يُصدره صاحب الخادم لكل مستخدم عبر backend/tools/issue_token.py). يُرسل في ترويسة Authorization
     * فقط، لا في الرابط. يُحفظ في تفضيلات التطبيق الخاصة (غير مشفّرة) ولا يدخل النسخ الاحتياطي (allowBackup=false).
     */
    fun getAccessToken(ctx: Context): String =
        ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(KEY_TOKEN, "")?.trim() ?: ""

    fun saveAccessToken(ctx: Context, token: String) {
        ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_TOKEN, token.trim())
            .apply()
    }

    fun normalizeBaseUrl(url: String): String {
        var u = url.trim().trimEnd('/')
        if (u.isEmpty()) return BACKEND_BASE_URL
        if (u.startsWith("wss://")) u = "https://" + u.removePrefix("wss://")
        if (u.startsWith("ws://")) u = "http://" + u.removePrefix("ws://")
        if (!u.startsWith("http://") && !u.startsWith("https://")) u = "https://$u"
        u = u.removeSuffix("/ws/live-recitation").removeSuffix("/health").trimEnd('/')
        return u
    }

    fun webSocketUrl(ctx: Context, strictness: String): String {
        val base = getServerBaseUrl(ctx)
        val wsBase = when {
            base.startsWith("https://") -> "wss://" + base.removePrefix("https://")
            base.startsWith("http://") -> "ws://" + base.removePrefix("http://")
            else -> "wss://$base"
        }
        return "$wsBase/ws/live-recitation?sura_idx=$SURA_IDX&aya_idx=$AYA_IDX&strictness=$strictness"
    }

    /** Modal bills per second of server uptime: do not wake the server just because the app opened. */
    fun isMetered(ctx: Context): Boolean = getServerBaseUrl(ctx).contains("modal.run")

    fun healthUrl(ctx: Context): String = getServerBaseUrl(ctx) + "/health"

    fun debugUrl(ctx: Context): String = getServerBaseUrl(ctx) + "/debug/last?steps=30"
}
