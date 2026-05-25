package com.solo.barbersmirror

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlin.math.hypot

class StaticFaceAnalyzer(context: Context) {

    private var faceLandmarker: FaceLandmarker? = null

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        // THE PIVOT: Configured strictly for static IMAGE processing
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .setErrorListener { error -> println("STATIC ENGINE ERROR: ${error.message}") }
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(context, options)
    }

    fun analyzeImage(bitmap: Bitmap): Pair<String, String>? {
        val mpImage = BitmapImageBuilder(bitmap).build()

        // detect() is synchronous. It runs instantly and returns the result, no listeners needed.
        val result = faceLandmarker?.detect(mpImage)

        if (result != null && result.faceLandmarks().isNotEmpty()) {
            return calculateFaceShape(result.faceLandmarks()[0], bitmap.width, bitmap.height)
        }
        return null
    }

    // Identical geometry matrix to your live feed
    private fun calculateFaceShape(landmarks: List<NormalizedLandmark>, width: Int, height: Int): Pair<String, String> {
        val topForehead = landmarks[10]
        val bottomChin = landmarks[152]
        val leftCheek = landmarks[234]
        val rightCheek = landmarks[454]
        val leftJaw = landmarks[148]
        val rightJaw = landmarks[377]

        val chinY = bottomChin.y() * height
        val topY = topForehead.y() * height
        val chinX = bottomChin.x() * width
        val topX = topForehead.x() * width

        val lCheekX = leftCheek.x() * width
        val rCheekX = rightCheek.x() * width
        val lCheekY = leftCheek.y() * height
        val rCheekY = rightCheek.y() * height

        val lJawX = leftJaw.x() * width
        val rJawX = rightJaw.x() * width
        val lJawY = leftJaw.y() * height
        val rJawY = rightJaw.y() * height

        val faceLength = hypot((chinX - topX).toDouble(), (chinY - topY).toDouble())
        val cheekWidth = hypot((rCheekX - lCheekX).toDouble(), (rCheekY - lCheekY).toDouble())
        val jawWidth = hypot((rJawX - lJawX).toDouble(), (rJawY - lJawY).toDouble())

        val lengthRatio = cheekWidth / faceLength
        val jawToCheekRatio = jawWidth / cheekWidth

        val telemetryStr = "STATIC L-Ratio: %.2f | J-Ratio: %.2f".format(lengthRatio, jawToCheekRatio)

        val shape = when {
            lengthRatio < 0.70 -> "OBLONG"
            jawToCheekRatio > 0.78 -> "SQUARE"
            lengthRatio > 0.85 -> "ROUND"
            else -> "OVAL"
        }

        return Pair(shape, telemetryStr)
    }
}