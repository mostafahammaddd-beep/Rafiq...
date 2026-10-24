package com.rafiq.app

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily

// آية الكرسي كاملة (البقرة 255) — 50 كلمة بالتشكيل الكامل
val WORDS: List<String> = listOf(
    "اللَّهُ", "لَا", "إِلَٰهَ", "إِلَّا", "هُوَ", "الْحَيُّ", "الْقَيُّومُ",
    "لَا", "تَأْخُذُهُ", "سِنَةٌ", "وَلَا", "نَوْمٌ",
    "لَهُ", "مَا", "فِي", "السَّمَاوَاتِ", "وَمَا", "فِي", "الْأَرْضِ",
    "مَنْ", "ذَا", "الَّذِي", "يَشْفَعُ", "عِنْدَهُ", "إِلَّا", "بِإِذْنِهِ",
    "يَعْلَمُ", "مَا", "بَيْنَ", "أَيْدِيهِمْ", "وَمَا", "خَلْفَهُمْ",
    "وَلَا", "يُحِيطُونَ", "بِشَيْءٍ", "مِنْ", "عِلْمِهِ", "إِلَّا", "بِمَا", "شَاءَ",
    "وَسِعَ", "كُرْسِيُّهُ", "السَّمَاوَاتِ", "وَالْأَرْضَ",
    "وَلَا", "يَئُودُهُ", "حِفْظُهُمَا", "وَهُوَ", "الْعَلِيُّ", "الْعَظِيمُ"
)

// تلاوة مرجعية (الشيخ المنشاوي) للاستماع قبل القراءة
const val AUDIO_URL = "https://everyayah.com/data/Minshawy_Murattal_128kbps/002255.mp3"

// ألوان الواجهة
val Bg = Color(0xFF0F2B2A)
val CardC = Color(0xFF173E3B)
val Gold = Color(0xFFD4AF37)
val Cream = Color(0xFFF5EBD0)
val Cairo: FontFamily = FontFamily.Serif

// ألوان نتيجة الكلمات
val GreenOk = Color(0xFF4ADE80)          // صحيحة
val YellowTashkeel = Color(0xFFFACC15)   // خطأ في التشكيل (الحركات)
val OrangeTajweed = Color(0xFFFF922B)    // خطأ في أحكام التجويد
val RedLetters = Color(0xFFEF4444)       // خطأ في الحروف / كلمة متروكة

fun statusColor(status: String?): Color = when (status) {
    "ok" -> GreenOk
    "tashkeel" -> YellowTashkeel
    "tajweed" -> OrangeTajweed
    "letters" -> RedLetters
    else -> Cream
}

fun statusTitle(type: String): String = when (type) {
    "tashkeel" -> "خطأ في التشكيل (الحركات)"
    "tajweed" -> "خطأ في أحكام التجويد"
    else -> "خطأ في الحروف"
}
