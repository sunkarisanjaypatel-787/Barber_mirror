package com.solo.barbersmirror

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

data class ClassificationResult(
    val shape: String,
    val lRatio: Float,
    val fRatio: Float,
    val gRatio: Float,
    val mRatio: Float
)

object GoldenVectorEngine {

    init {
        System.loadLibrary("golden_vector")
    }

    private external fun predictShapeNative(h_w: Float, j_w: Float, f_j: Float, c_j: Float): String

    private val SHAPES = arrayOf("oblong", "oval", "round", "square", "triangle")

    private fun calcDist(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        val dx = p1.x() - p2.x()
        val dy = p1.y() - p2.y()
        val dz = p1.z() - p2.z()
        return sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2))
    }

    private fun calcDist2D(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        val dx = p1.x() - p2.x()
        val dy = p1.y() - p2.y()
        return sqrt(dx.pow(2) + dy.pow(2))
    }

    fun isFaceAligned(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size < 468) return false
        val noseToLeftCheek = calcDist2D(landmarks[1], landmarks[234])
        val noseToRightCheek = calcDist2D(landmarks[1], landmarks[454])
        val yawRatio = if (noseToRightCheek > 0) noseToLeftCheek / noseToRightCheek else 0f

        val noseToChin = calcDist2D(landmarks[1], landmarks[152])
        val noseToForehead = calcDist2D(landmarks[1], landmarks[10])
        val pitchRatio = if (noseToForehead > 0) noseToChin / noseToForehead else 0f

        val cheekDepthDiff = abs(landmarks[234].z() - landmarks[454].z())
        val isZLocked = cheekDepthDiff < 0.04f

        return (yawRatio in 0.85f..1.15f) && (pitchRatio in 0.80f..1.20f) && isZLocked
    }

    fun analyzeFaceShape(landmarks: List<NormalizedLandmark>): ClassificationResult {
        if (landmarks.size < 468) return ClassificationResult("UNDETERMINED", 0f, 0f, 0f, 0f)

        // Extract the 4D matrix directly from the 3D mesh
        val faceHeight = calcDist(landmarks[10], landmarks[152])
        val faceWidth = calcDist(landmarks[234], landmarks[454])
        val jawWidth = calcDist(landmarks[132], landmarks[361])
        val foreheadWidth = calcDist(landmarks[54], landmarks[284])

        val h_w = if (faceWidth > 0) faceHeight / faceWidth else 0f
        val j_w = if (faceWidth > 0) jawWidth / faceWidth else 0f
        val f_j = if (jawWidth > 0) foreheadWidth / jawWidth else 0f
        val c_j = if (jawWidth > 0) faceWidth / jawWidth else 0f

        // Feed the ratios into the native C++ decision trees via JNI
        val shape = predictShapeNative(h_w, j_w, f_j, c_j)

        return ClassificationResult(shape.uppercase(), h_w, j_w, f_j, c_j)
    }
}
