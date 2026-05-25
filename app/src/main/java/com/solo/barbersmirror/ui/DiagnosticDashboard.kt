package com.solo.barbersmirror.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.solo.barbersmirror.AccuracyBenchmark
import com.solo.barbersmirror.BenchmarkStats
import kotlinx.coroutines.launch

@Composable
fun DiagnosticDashboard(onClose: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val benchmark = remember { AccuracyBenchmark(context) }
    var stats by remember { mutableStateOf(BenchmarkStats()) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0A))
            .padding(16.dp)
    ) {
        // --- HEADER ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "ENGINE_DIAGNOSTICS // V1.0",
                color = Color(0xFF00FFCC),
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            )
            IconButton(onClick = onClose) {
                Text("X", color = Color.Red, fontWeight = FontWeight.Bold)
            }
        }

        HorizontalDivider(color = Color(0xFF00FFCC).copy(alpha = 0.3f), modifier = Modifier.padding(vertical = 8.dp))

        // --- MAIN METRICS ---
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A1A)),
            modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("OVERALL_ACCURACY", color = Color.Gray, fontSize = 10.sp)
                Text(
                    "${"%.1f".format(stats.accuracyScore)}%",
                    color = if (stats.accuracyScore > 80) Color.Green else Color.Yellow,
                    fontSize = 48.sp,
                    fontWeight = FontWeight.Black
                )
                Text("TOTAL: ${stats.totalProcessed}", color = Color.Gray, fontSize = 12.sp)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("CORRECT: ${stats.correctGuesses}", color = Color.Green, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Text("WRONG: ${stats.wrongGuesses}", color = Color.Red, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Text("RE-ALIGN: ${stats.reAlignCount}", color = Color.Yellow, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                }

                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = Color.DarkGray)
                Spacer(modifier = Modifier.height(8.dp))

                // --- SHAPE-WISE BREAKDOWN ---
                Text("SHAPE_PERFORMANCE:", color = Color.Gray, fontSize = 10.sp)
                stats.confusionMatrix.keys.sorted().forEach { shape ->
                    val row = stats.confusionMatrix[shape] ?: emptyMap()
                    val totalForShape = row.values.sum()
                    val correctForShape = row[shape] ?: 0
                    val acc = if (totalForShape > 0) (correctForShape.toFloat() / totalForShape) * 100 else 0f
                    
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(shape, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                        Text("${"%.1f".format(acc)}% ($correctForShape/$totalForShape)", 
                            color = if (acc > 80) Color.Green else if (acc > 50) Color.Yellow else Color.Red,
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }

        // --- CONTROLS ---
        Button(
            onClick = {
                scope.launch {
                    benchmark.runStressTest { updatedStats ->
                        stats = updatedStats
                    }
                }
            },
            enabled = !stats.isRunning,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = if (stats.isRunning) Color.DarkGray else Color(0xFF00FFCC)),
            shape = RoundedCornerShape(4.dp)
        ) {
            Text(
                if (stats.isRunning) "ANALYSING_DATA_STREAM..." else "INITIATE_STRESS_TEST",
                color = if (stats.isRunning) Color.Gray else Color.Black,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // --- CONFUSION MATRIX & FAILURES ---
        Text("SYSTEM_ERROR_LOG / FAILURES", color = Color.Gray, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp))
        
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .border(1.dp, Color.White.copy(alpha = 0.1f))
                .background(Color.Black)
        ) {
            items(stats.failures) { failure ->
                FailureItem(failure)
            }
        }
    }
}

@Composable
fun FailureItem(failure: com.solo.barbersmirror.BenchmarkFailure) {
    val context = LocalContext.current
    val bitmap = remember(failure.imagePath) {
        try {
            val stream = context.assets.open(failure.imagePath)
            BitmapFactory.decodeStream(stream)
        } catch (_: Exception) {
            null
        }
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .border(1.dp, Color.Red.copy(alpha = 0.2f), RoundedCornerShape(4.dp))
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.size(60.dp).background(Color.DarkGray)
            )
        }
        
        Spacer(modifier = Modifier.width(12.dp))
        
        Column {
            Text("EXPECTED: ${failure.groundTruth}", color = Color.Green, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text("PREDICTED: ${failure.predicted}", color = Color.Red, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text("RATIOS: L=${"%.2f".format(failure.lRatio)} F=${"%.2f".format(failure.fRatio)} G=${"%.2f".format(failure.gRatio)} M=${"%.2f".format(failure.mRatio)}", color = Color.Yellow, fontSize = 9.sp)
            Text(failure.imagePath, color = Color.DarkGray, fontSize = 9.sp)
        }
    }
}
