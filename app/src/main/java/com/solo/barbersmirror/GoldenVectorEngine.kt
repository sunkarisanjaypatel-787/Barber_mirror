package com.solo.barbersmirror

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.collections.iterator
import kotlin.math.pow
import kotlin.math.sqrt

object GoldenVectorEngine {

    // 1. THE EXPANDED ANATOMICAL VAULT
    // We now include "Tall" and "Short" variants of the shapes to catch edge cases.
    private val REFERENCE_VAULT = listOf(
        // SQUARE: Wide lower jaw (>0.73)
        Pair("SQUARE", floatArrayOf(1.15f, 0.70f, 0.95f, 0.74f, 0.24f)), // Standard Short Square
        Pair("SQUARE", floatArrayOf(1.25f, 0.70f, 0.94f, 0.74f, 0.24f)), // TALL Square (Cavill Archetype)
        Pair("SQUARE", floatArrayOf(1.12f, 0.68f, 0.93f, 0.73f, 0.24f)),

        // ROUND: Narrower lower jaw (~0.70), Short face
        Pair("ROUND", floatArrayOf(1.15f, 0.70f, 0.90f, 0.70f, 0.25f)),
        Pair("ROUND", floatArrayOf(1.20f, 0.72f, 0.91f, 0.71f, 0.25f)),  // Slightly taller Round
        Pair("ROUND", floatArrayOf(1.12f, 0.68f, 0.88f, 0.69f, 0.25f)),

        // OVAL: Tall face, Moderate lower jaw
        Pair("OVAL", floatArrayOf(1.35f, 0.70f, 0.90f, 0.71f, 0.25f)),   // Standard Tall Oval
        Pair("OVAL", floatArrayOf(1.25f, 0.70f, 0.91f, 0.71f, 0.25f)),   // SHORT Oval (Edge case anchor)
        Pair("OVAL", floatArrayOf(1.32f, 0.68f, 0.88f, 0.70f, 0.25f)),

        // OBLONG: Extremely tall face, Narrowest jaw (<0.70)
        Pair("OBLONG", floatArrayOf(1.45f, 0.70f, 0.92f, 0.69f, 0.25f)),
        Pair("OBLONG", floatArrayOf(1.40f, 0.70f, 0.91f, 0.69f, 0.25f)),
        Pair("OBLONG", floatArrayOf(1.48f, 0.72f, 0.94f, 0.70f, 0.25f))
    )

    // 3D Euclidean Distance Calculator
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

        // 1. YAW (Left/Right Turn) -> Must use 2D projection to catch width compression
        val noseToLeftCheek = calcDist2D(landmarks[1], landmarks[234])
        val noseToRightCheek = calcDist2D(landmarks[1], landmarks[454])
        val yawRatio = if (noseToRightCheek > 0) noseToLeftCheek / noseToRightCheek else 0f

        // 2. PITCH (Up/Down Tilt) -> Must use 2D projection to catch height compression
        val noseToChin = calcDist2D(landmarks[1], landmarks[152])
        val noseToForehead = calcDist2D(landmarks[1], landmarks[10])
        val pitchRatio = if (noseToForehead > 0) noseToChin / noseToForehead else 0f

        // 3. ROLL & DEPTH (Z-axis Check)
        // If one cheek is closer to the camera than the other, the Z difference spikes
        val cheekDepthDiff = Math.abs(landmarks[234].z() - landmarks[454].z())
        val isZLocked = cheekDepthDiff < 0.04f

        // Tolerances calibrated for 2D compression
        val isYawLocked = yawRatio in 0.85f..1.15f
        val isPitchLocked = pitchRatio in 0.80f..1.20f

        return isYawLocked && isPitchLocked && isZLocked
    }

    // 2. THE MULTI-VECTOR CLUSTERING ENGINE (KNN)
    fun analyzeFaceShape(landmarks: List<NormalizedLandmark>): ClassificationResult {
        if (landmarks.size < 468) return ClassificationResult("UNDETERMINED", 0f, 0f, 0f, 0f)

        // --- THE QUALITY GUARD: POSE & BOUNDARY CHECK ---
        // Ensure the face isn't too close or cut off (Normalized coordinates should be 0.1 to 0.9)
        val centerPoint = landmarks[1]
        val isOffCenter = centerPoint.x() < 0.1f || centerPoint.x() > 0.9f
        val isTooClose = calcDist(landmarks[234], landmarks[454]) > 0.8f

        // Check for extreme head tilt (Yaw/Roll) using lateral symmetry
        val leftEye = landmarks[33]
        val rightEye = landmarks[263]
        val eyeLevelDifference = Math.abs(leftEye.y() - rightEye.y())

        if (isOffCenter || isTooClose || eyeLevelDifference > 0.15f) {
            return ClassificationResult("RE-ALIGN FACE", 0f, 0f, 0f, 0f)
        }

        val cheekW = calcDist(landmarks[234], landmarks[454])
        if (cheekW == 0f) return ClassificationResult("UNDETERMINED", 0f, 0f, 0f, 0f)

        val faceL = calcDist(landmarks[10], landmarks[152])
        val foreheadW = calcDist(landmarks[103], landmarks[332])
        val upperJawW = calcDist(landmarks[132], landmarks[361])
        val lowerJawW = calcDist(landmarks[136], landmarks[365])

        val liveVector = floatArrayOf(
            faceL / cheekW,
            foreheadW / cheekW,
            upperJawW / cheekW,
            lowerJawW / cheekW,
            (upperJawW - lowerJawW) / cheekW
        )

        // PROTOCOL ALPHA: The Balanced Master Weights
        val weights = floatArrayOf(
            2.0f,  // [0] Aspect Ratio: High enough to separate Round/Oblong, low enough to allow Tall Squares.
            0.5f,  // [1] Forehead Ratio: Minimal impact.
            2.0f,  // [2] Upper Jaw: Crucial for the Oval cheekbone curve.
            6.0f,  // [3] Lower Jaw: Absolute maximum authority. Jaw width defines the Square.
            0.0f   // [4] Taper Delta: Kept dead.
        )

        val distances = mutableListOf<Pair<String, Float>>()

        for (reference in REFERENCE_VAULT) {
            val shapeLabel = reference.first
            val refVector = reference.second

            var distanceSum = 0f
            for (i in 0..4) {
                val liveScaled = liveVector[i] * 100f
                val refScaled = refVector[i] * 100f
                distanceSum += weights[i] * (liveScaled - refScaled).pow(2)
            }
            distances.add(Pair(shapeLabel, sqrt(distanceSum)))
        }

        // PROTOCOL BETA: K-Nearest Neighbors (Find the top 3 closest matches)
        distances.sortBy { it.second }
        val top3Matches = distances.take(3)

        // Tally the votes
        val voteCount = mutableMapOf<String, Int>()
        for (match in top3Matches) {
            voteCount[match.first] = voteCount.getOrDefault(match.first, 0) + 1
        }

        // The shape with the most votes wins the classification
        val finalShape = voteCount.maxByOrNull { it.value }?.key ?: "UNDETERMINED"

        return ClassificationResult(
            finalShape,
            liveVector[0], liveVector[1], liveVector[2], liveVector[3]
        )
    }
}

data class ClassificationResult(
    val shape: String,
    val lRatio: Float,
    val fRatio: Float,
    val gRatio: Float,
    val mRatio: Float
)
