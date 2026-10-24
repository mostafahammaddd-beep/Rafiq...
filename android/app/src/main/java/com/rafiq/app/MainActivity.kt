package com.rafiq.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.onClick
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlin.math.max

class MainActivity : ComponentActivity() {
    private var controller: RafiqController? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val c = RafiqController(applicationContext)
        controller = c
        // لا إيقاظ للخادم عند فتح التطبيق: لا حاوية تُفوتَر أثناء التصفح. تبدأ المصافحة مع الخادم لحظة لمس زر التسجيل.
        c.markSleeping()
        setContent {
            CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                RafiqApp(c)
            }
        }
    }

    override fun onStop() {
        controller?.cancelIfActive()
        super.onStop()
    }

    override fun onDestroy() {
        controller?.release()
        super.onDestroy()
    }
}

@Composable
fun RafiqApp(c: RafiqController) {
    val ctx = LocalContext.current
    val view = LocalView.current
    SideEffect { view.keepScreenOn = (c.stage == Stage.LISTENING || c.stage == Stage.STARTING) }

    var selected by remember { mutableStateOf<Int?>(null) }
    var showSettings by remember { mutableStateOf(false) }

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) c.startRecitation()
    }

    val onMic: () -> Unit = {
        if (c.stage == Stage.LISTENING) {
            c.stopRecitation()
        } else if (c.stage == Stage.STARTING) {
            c.cancelStart()
        } else if (c.stage != Stage.FINISHING) {
            if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
            ) {
                c.startRecitation()
            } else {
                permLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    Box(Modifier.fillMaxSize().background(Bg)) {
        Column(
            Modifier
                .fillMaxSize()
                .padding(horizontal = 14.dp, vertical = 12.dp)
        ) {
            TopBar(c, onSettings = { showSettings = true })
            Spacer(Modifier.height(8.dp))
            Box(Modifier.weight(1f).fillMaxWidth()) {
                VerseCard(c, onWordClick = { selected = it })
            }
            Spacer(Modifier.height(8.dp))
            Legend()
            Spacer(Modifier.height(6.dp))
            BottomArea(c, onMic)
        }
    }

    val sel = selected
    if (sel != null) {
        val r = c.results.getOrNull(sel)
        if (r != null) {
            WordDialog(
                word = c.words.getOrElse(sel) { "" },
                r = r,
                onDismiss = { selected = null },
                onReport = { kind -> c.reportWord(sel, kind) }
            )
        }
    }
    if (showSettings) {
        SettingsDialog(c, onDismiss = { showSettings = false })
    }
}

@Composable
fun TopBar(c: RafiqController, onSettings: () -> Unit) {
    val statusText: String
    val chipColor: Color
    when {
        c.serverState == 1 -> {
            statusText = "● الخادم جاهز"
            chipColor = GreenOk
        }
        c.serverState == 2 -> {
            statusText = "⚠ لا اتصال بالخادم (اضغط)"
            chipColor = RedLetters
        }
        c.serverState == 3 -> {
            statusText = "⏸ الخادم نائم — يستيقظ لحظة الضغط على التسجيل"
            chipColor = Cream
        }
        else -> {
            statusText = "⏳ تجهيز الخادم…"
            chipColor = Gold
        }
    }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(
                "رَفِيق",
                color = Gold,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = Cairo
            )
            Text(
                "آية الكرسي — سورة البقرة ٢٥٥",
                color = Cream.copy(alpha = 0.75f),
                fontSize = 12.sp
            )
        }
        Text(
            statusText,
            color = chipColor,
            fontSize = 12.sp,
            modifier = Modifier
                .clip(RoundedCornerShape(50))
                .clickable(enabled = c.serverState == 2 || c.serverState == 3) { c.retryServer() }
                .padding(horizontal = 8.dp, vertical = 4.dp)
        )
        Spacer(Modifier.width(4.dp))
        Text(
            "⚙",
            color = Cream,
            fontSize = 24.sp,
            modifier = Modifier
                .clip(CircleShape)
                .clickable { onSettings() }
                .padding(6.dp)
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun VerseCard(c: RafiqController, onWordClick: (Int) -> Unit) {
    val scroll = rememberScrollState()
    val ys = remember { HashMap<Int, Float>() }
    val pulse by rememberInfiniteTransition(label = "pulse").animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(750), RepeatMode.Reverse),
        label = "pulseAlpha"
    )

    LaunchedEffect(c.glow) {
        if (c.stage == Stage.LISTENING) {
            val y = ys[c.glow]
            if (y != null) scroll.animateScrollTo(max(0, (y - 140f).toInt()))
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .clip(RoundedCornerShape(20.dp))
            .background(CardC)
            .verticalScroll(scroll)
            .padding(14.dp)
    ) {
        FlowRow(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            c.words.forEachIndexed { i, w ->
                WordChip(
                    text = w,
                    status = c.results.getOrNull(i)?.status,
                    active = (c.stage == Stage.LISTENING && i == c.glow),
                    pulse = pulse,
                    onClick = { onWordClick(i) },
                    onPlaced = { y -> ys[i] = y }
                )
            }
        }
    }
}

@Composable
fun WordChip(
    text: String,
    status: String?,
    active: Boolean,
    pulse: Float,
    onClick: () -> Unit,
    onPlaced: (Float) -> Unit
) {
    val color = when (status) {
        "ok" -> GreenOk
        "tashkeel" -> YellowTashkeel
        "tajweed" -> OrangeTajweed
        "letters" -> RedLetters
        "unchecked" -> Cream.copy(alpha = 0.5f)
        else -> if (active) Gold else Cream.copy(alpha = 0.92f)
    }
    val isError = status == "tashkeel" || status == "tajweed" || status == "letters"
    val bg = when {
        active -> Gold.copy(alpha = 0.10f + 0.22f * pulse)
        isError -> color.copy(alpha = 0.14f)
        else -> Color.Transparent
    }
    val shape = RoundedCornerShape(10.dp)
    Box(
        modifier = Modifier
            .onGloballyPositioned { onPlaced(it.positionInParent().y) }
            .clip(shape)
            .background(bg)
            .then(if (active) Modifier.border(1.dp, Gold.copy(alpha = 0.7f), shape) else Modifier)
            .clickable(enabled = status != null && status != "unchecked") { onClick() }
            .padding(horizontal = 6.dp, vertical = 2.dp)
    ) {
        Text(
            text = text,
            color = color,
            fontSize = 27.sp,
            fontFamily = Cairo,
            fontWeight = if (isError || active) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
fun Legend() {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        LegendItem(GreenOk, "صحيح")
        LegendItem(YellowTashkeel, "تشكيل")
        LegendItem(OrangeTajweed, "تجويد")
        LegendItem(RedLetters, "حروف")
    }
}

@Composable
fun LegendItem(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(5.dp))
        Text(label, color = Cream.copy(alpha = 0.85f), fontSize = 12.sp)
    }
}

@Composable
fun BottomArea(c: RafiqController, onMic: () -> Unit) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        if (c.stage == Stage.DONE) {
            ResultCard(c)
        }
        if (c.message.isNotEmpty()) {
            Text(
                c.message,
                color = Gold,
                fontSize = 13.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(vertical = 4.dp)
            )
        } else if (c.stage == Stage.IDLE) {
            Text(
                "اضغط على الميكروفون وابدأ التلاوة من أول الآية",
                color = Cream.copy(alpha = 0.8f),
                fontSize = 13.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(vertical = 4.dp)
            )
        }
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            Box(Modifier.align(Alignment.CenterStart)) {
                val busyOrListening = c.stage == Stage.LISTENING || c.stage == Stage.FINISHING || c.stage == Stage.STARTING
                if (!busyOrListening) {
                    Text(
                        if (c.playing) "⏹ إيقاف" else "🔊 استمع",
                        color = Gold,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .border(1.dp, Gold.copy(alpha = 0.6f), RoundedCornerShape(50))
                            .clickable { c.toggleReference() }
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                    )
                }
            }
            MicButton(c, onMic)
            Box(Modifier.align(Alignment.CenterEnd)) {
                if (c.stage == Stage.LISTENING) {
                    Text(
                        String.format("%d:%02d", c.seconds / 60, c.seconds % 60),
                        color = Cream.copy(alpha = 0.8f),
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
fun MicButton(c: RafiqController, onMic: () -> Unit) {
    val listening = c.stage == Stage.LISTENING
    val busy = c.stage == Stage.FINISHING
    val starting = c.stage == Stage.STARTING
    val ring by animateFloatAsState(
        targetValue = if (listening) (c.level * 10f).coerceIn(0f, 1f) else 0f,
        label = "ring"
    )
    Box(contentAlignment = Alignment.Center, modifier = Modifier.size(112.dp)) {
        if (listening) {
            Box(
                Modifier
                    .size((84f + 26f * ring).dp)
                    .clip(CircleShape)
                    .background(RedLetters.copy(alpha = 0.25f))
            )
        }
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(84.dp)
                .clip(CircleShape)
                .background(if (listening) RedLetters else Gold)
                // ACTION_DOWN: المصافحة المبكرة مع الخادم تبدأ لحظة اللمس لا لحظة الرفع (وتوفّر ~100-200ms من الانتظار)
                .pointerInput(busy) {
                    detectTapGestures(onPress = { if (!busy) onMic() })
                }
                .semantics { onClick(label = "تسجيل / إيقاف") { if (!busy) onMic(); true } }
        ) {
            if (busy || starting) {
                CircularProgressIndicator(
                    color = Bg,
                    strokeWidth = 3.dp,
                    modifier = Modifier.size(34.dp)
                )
            } else {
                Text(if (listening) "⏹" else "🎤", fontSize = 34.sp)
            }
        }
    }
}

@Composable
fun ResultCard(c: RafiqController) {
    val s = c.summary()
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(CardC)
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (s.judged > 0) {
            Text(
                "أتقنتَ ${s.ok} من ${s.judged} كلمة",
                color = if (s.ok == s.judged) GreenOk else Gold,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(6.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                if (s.letters > 0) CountChip(RedLetters, "حروف", s.letters)
                if (s.tashkeel > 0) CountChip(YellowTashkeel, "تشكيل", s.tashkeel)
                if (s.tajweed > 0) CountChip(OrangeTajweed, "تجويد", s.tajweed)
            }
            if (s.letters + s.tashkeel + s.tajweed > 0) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "اضغط على أي كلمة ملوّنة لتعرف الخطأ وطريقة تصحيحه",
                    color = Cream.copy(alpha = 0.85f),
                    fontSize = 13.sp,
                    textAlign = TextAlign.Center
                )
            } else if (s.unchecked == 0 && s.judged >= s.total) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "ما شاء الله! تلاوة سليمة 🌟",
                    color = GreenOk,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            if (s.judged < s.total) {
                Spacer(Modifier.height(4.dp))
                Text(
                    "لم تُكمل الآية؛ تم الحكم على ${s.judged} كلمة من ${s.total}.",
                    color = Cream.copy(alpha = 0.7f),
                    fontSize = 12.sp,
                    textAlign = TextAlign.Center
                )
            }
        } else {
            Text("لم يتم رصد تلاوة.", color = Cream, fontSize = 15.sp)
        }
    }
    Spacer(Modifier.height(6.dp))
}

@Composable
fun CountChip(color: Color, label: String, n: Int) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(5.dp))
        Text("$label: $n", color = Cream, fontSize = 13.sp)
    }
}

@Composable
fun WordDialog(word: String, r: WordResult, onDismiss: () -> Unit, onReport: (String) -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = CardC,
        titleContentColor = Gold,
        textContentColor = Cream,
        title = {
            Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    word,
                    color = statusColor(r.status),
                    fontSize = 34.sp,
                    fontFamily = Cairo,
                    fontWeight = FontWeight.Bold
                )
            }
        },
        text = {
            Column(
                Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (r.status == "ok") {
                    Text(
                        "✔ أحسنت! نطق هذه الكلمة صحيح.",
                        color = GreenOk,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                } else if (r.issues.isEmpty()) {
                    Text(
                        "لم تُفحص هذه الكلمة بدقة لأن الاتصال كان بطيئًا.",
                        color = Cream,
                        fontSize = 15.sp
                    )
                } else {
                    r.issues.forEach { iss -> IssueBlock(iss) }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("حسنًا", color = Gold, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            if (r.status == "ok") {
                TextButton(onClick = {
                    onReport("missed")
                    onDismiss()
                }) {
                    Text("⚠ فيها خطأ لم يُكتشف", color = Cream, fontSize = 12.sp)
                }
            } else if (r.status != "unchecked") {
                TextButton(onClick = {
                    onReport("wrong_flag")
                    onDismiss()
                }) {
                    Text("✔ نطقتها صحيحة", color = Cream, fontSize = 12.sp)
                }
            }
        }
    )
}

@Composable
fun IssueBlock(iss: WordIssue) {
    val col = statusColor(iss.type)
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(col.copy(alpha = 0.13f))
            .padding(10.dp)
    ) {
        Text(statusTitle(iss.type), color = col, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Text(iss.note, color = Cream, fontSize = 15.sp, lineHeight = 24.sp)
    }
}

@Composable
fun StrictChip(label: String, value: String, c: RafiqController, modifier: Modifier) {
    val selected = c.strictness == value
    val shape = RoundedCornerShape(10.dp)
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .clip(shape)
            .background(if (selected) Gold else Bg)
            .border(1.dp, Gold.copy(alpha = 0.6f), shape)
            .clickable { c.changeStrictness(value) }
            .padding(vertical = 8.dp)
    ) {
        Text(
            label,
            color = if (selected) Bg else Cream,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun SettingsDialog(c: RafiqController, onDismiss: () -> Unit) {
    val ctx = LocalContext.current
    var urlInput by remember { mutableStateOf(RecitationConfig.getServerBaseUrl(ctx)) }
    val strictnessHelp = when (c.strictness) {
        "relaxed" -> "يتجاوز الهفوات الصغيرة — مناسب للمبتدئين."
        "strict" -> "يلتقط أدق الفروق، وقد ينبّه على هفوات بسيطة."
        else -> "الأنسب لمعظم القرّاء."
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = CardC,
        titleContentColor = Gold,
        textContentColor = Cream,
        title = { Text("الإعدادات", fontSize = 18.sp, fontWeight = FontWeight.Bold) },
        text = {
            Column(
                Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("درجة الدقة", color = Gold, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    StrictChip("متساهل", "relaxed", c, Modifier.weight(1f))
                    StrictChip("متوازن", "balanced", c, Modifier.weight(1f))
                    StrictChip("صارم", "strict", c, Modifier.weight(1f))
                }
                Text(strictnessHelp, color = Cream.copy(alpha = 0.85f), fontSize = 12.sp)
                Text("رابط الخادم", color = Gold, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                OutlinedTextField(
                    value = urlInput,
                    onValueChange = { urlInput = it },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Cream,
                        unfocusedTextColor = Cream,
                        focusedBorderColor = Gold,
                        unfocusedBorderColor = Cream.copy(alpha = 0.5f),
                        cursorColor = Gold
                    ),
                    modifier = Modifier.fillMaxWidth()
                )
                TextButton(onClick = { urlInput = RecitationConfig.BACKEND_BASE_URL }) {
                    Text("استعادة الرابط الافتراضي", color = Gold, fontSize = 12.sp)
                }
                OutlinedButton(
                    onClick = {
                        c.copyDebugReport()
                        onDismiss()
                    },
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Gold)
                ) {
                    Text("نسخ تقرير فني (للمطوّر)", fontSize = 13.sp)
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (urlInput.trim() != RecitationConfig.getServerBaseUrl(ctx)) {
                        c.saveServerUrl(urlInput)
                    }
                    onDismiss()
                },
                colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Bg)
            ) {
                Text("حفظ", fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("إغلاق", color = Cream)
            }
        }
    )
}
