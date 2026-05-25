package com.solo.barbersmirror

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.junit.Test
import org.junit.Assert.*

class EngineDiagnosticTest {

    // Helper to create a mock landmark
    private fun mockLM(x: Float, y: Float, z: Float) = NormalizedLandmark.create(x, y, z)

    @Test
    fun testOblongDetection() {
        // Constructing a mathematically LONG face (High Length/Width ratio)
        // Cheek Width (234 to 454) = 0.4
        // Face Length (10 to 152) = 0.8
        // Ratio = 2.0 (Should definitely be OBLONG)
        
        val mesh = MutableList(468) { mockLM(0.5f, 0.5f, 0f) }
        
        // Quality Guard Bypass (Center the face)
        mesh[1] = mockLM(0.5f, 0.5f, 0f)
        mesh[33] = mockLM(0.4f, 0.4f, 0f)
        mesh[263] = mockLM(0.6f, 0.4f, 0f)
        
        // Setting Width
        mesh[234] = mockLM(0.3f, 0.5f, 0f)
        mesh[454] = mockLM(0.7f, 0.5f, 0f) // Width = 0.4
        
        // Setting Length
        mesh[10] = mockLM(0.5f, 0.1f, 0f)
        mesh[152] = mockLM(0.5f, 0.9f, 0f) // Length = 0.8
        
        // Ratios
        // L = 0.8 / 0.4 = 2.0
        
        val result = GoldenVectorEngine.analyzeFaceShape(mesh)
        
        println("DIAGNOSTIC: For a Long Face (L-Ratio=2.0), Engine predicted: ${result.shape}")
        
        assertEquals("OBLONG", result.shape)
    }

    @Test
    fun testRoundDetection() {
        val mesh = MutableList(468) { mockLM(0.5f, 0.5f, 0f) }
        mesh[1] = mockLM(0.5f, 0.5f, 0f)
        mesh[33] = mockLM(0.4f, 0.4f, 0f)
        mesh[263] = mockLM(0.6f, 0.4f, 0f)
        
        // Setting Width
        mesh[234] = mockLM(0.2f, 0.5f, 0f)
        mesh[454] = mockLM(0.8f, 0.5f, 0f) // Width = 0.6
        
        // Setting Length (Short for round)
        mesh[10] = mockLM(0.5f, 0.2f, 0f)
        mesh[152] = mockLM(0.5f, 0.8f, 0f) // Length = 0.6
        
        val result = GoldenVectorEngine.analyzeFaceShape(mesh)
        println("DIAGNOSTIC: For a Round/Wide Face, Engine predicted: ${result.shape}")
        assertNotEquals("UNDETERMINED", result.shape)
    }
}
