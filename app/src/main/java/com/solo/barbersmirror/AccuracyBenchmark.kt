package com.solo.barbersmirror

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStream

data class BenchmarkStats(
    val totalProcessed: Int = 0,
    val correctGuesses: Int = 0,
    val wrongGuesses: Int = 0,
    val reAlignCount: Int = 0,
    val accuracyScore: Float = 0f,
    val confusionMatrix: Map<String, Map<String, Int>> = emptyMap(),
    val failures: List<BenchmarkFailure> = emptyList(),
    val isRunning: Boolean = false
)

data class BenchmarkFailure(
    val imagePath: String,
    val groundTruth: String,
    val predicted: String,
    val telemetry: String,
    val lRatio: Float = 0f,
    val fRatio: Float = 0f,
    val gRatio: Float = 0f,
    val mRatio: Float = 0f
)

class AccuracyBenchmark(private val context: Context) {

    private var faceLandmarker: FaceLandmarker? = null

    init {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .build()
            faceLandmarker = FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            println("BENCHMARK INIT FAILED: ${e.message}")
        }
    }

    suspend fun runStressTest(onProgress: (BenchmarkStats) -> Unit) = withContext(Dispatchers.Default) {
        val assetManager = context.assets
        // The two possible folder naming conventions found in assets and others/
        val folders = listOf("oblong", "oval", "round", "square", "oblong_face", "oval_face", "round_face", "square_face")
        
        var total = 0
        var correct = 0
        var wrong = 0
        var reAlign = 0
        val confusion = mutableMapOf<String, MutableMap<String, Int>>()
        val failures = mutableListOf<BenchmarkFailure>()

        // Initialize confusion matrix for the 4 base shapes
        listOf("OBLONG", "OVAL", "ROUND", "SQUARE").forEach { gt ->
            confusion[gt] = mutableMapOf(
                "SQUARE" to 0, "ROUND" to 0, "OVAL" to 0, "OBLONG" to 0, "UNDETERMINED" to 0, "RE-ALIGN FACE" to 0, "NO_FACE" to 0
            )
        }

        folders.forEach { folder ->
            val groundTruth = folder.split("_")[0].uppercase()
            // If the folder is one of our targets, process its contents recursively
            if (groundTruth in confusion.keys) {
                processFolderRecursively(folder, groundTruth, assetManager) { result, predicted, imgPath ->
                    total++
                    confusion[groundTruth]?.set(predicted, (confusion[groundTruth]?.get(predicted) ?: 0) + 1)
                    
                    when (predicted) {
                        groundTruth -> correct++
                        "RE-ALIGN FACE" -> reAlign++
                        else -> wrong++
                    }

                    if (predicted != groundTruth) {
                        val failure = BenchmarkFailure(
                            imagePath = imgPath,
                            groundTruth = groundTruth,
                            predicted = predicted,
                            telemetry = "L: ${"%.2f".format(result.lRatio)} | F: ${"%.2f".format(result.fRatio)}",
                            lRatio = result.lRatio,
                            fRatio = result.fRatio,
                            gRatio = result.gRatio,
                            mRatio = result.mRatio
                        )
                        failures.add(failure)
                    }

                    onProgress(BenchmarkStats(
                        totalProcessed = total,
                        correctGuesses = correct,
                        wrongGuesses = wrong,
                        reAlignCount = reAlign,
                        accuracyScore = if (total > 0) (correct.toFloat() / total) * 100 else 0f,
                        confusionMatrix = confusion,
                        failures = failures,
                        isRunning = true
                    ))
                }
            }
        }
        
        onProgress(BenchmarkStats(
            totalProcessed = total,
            correctGuesses = correct,
            wrongGuesses = wrong,
            reAlignCount = reAlign,
            accuracyScore = if (total > 0) (correct.toFloat() / total) * 100 else 0f,
            confusionMatrix = confusion,
            failures = failures,
            isRunning = false
        ))
    }

    private fun processFolderRecursively(
        path: String,
        groundTruth: String,
        assetManager: android.content.res.AssetManager,
        onProcessed: (ClassificationResult, String, String) -> Unit
    ) {
        val items = assetManager.list(path) ?: return
        items.forEach { item ->
            val subPath = if (path.isEmpty()) item else "$path/$item"
            if (item.endsWith(".png") || item.endsWith(".jpg") || item.endsWith(".jpeg") || item.endsWith(".webp")) {
                val bitmap = loadBitmapFromAsset(subPath)
                if (bitmap != null) {
                    val result = analyzeBitmap(bitmap)
                    val predicted = result?.shape ?: "UNDETERMINED"
                    if (result != null) {
                        onProcessed(result, predicted, subPath)
                    }
                }
            } else if (!item.contains(".")) {
                // Likely a directory, go deeper
                processFolderRecursively(subPath, groundTruth, assetManager, onProcessed)
            }
        }
    }

    private fun loadBitmapFromAsset(path: String): Bitmap? {
        return try {
            val inputStream: InputStream = context.assets.open(path)
            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            null
        }
    }

    private fun analyzeBitmap(bitmap: Bitmap): ClassificationResult? {
        if (faceLandmarker == null) return null
        return try {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val mpImage = BitmapImageBuilder(argbBitmap).build()
            val result = faceLandmarker?.detect(mpImage)
            if (result != null && result.faceLandmarks().isNotEmpty()) {
                GoldenVectorEngine.analyzeFaceShape(result.faceLandmarks()[0])
            } else {
                ClassificationResult("NO_FACE", 0f, 0f, 0f, 0f)
            }
        } catch (e: Exception) {
            null
        }
    }
}
