package com.solo.barbersmirror

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.experimental.xor

/**
 * Lightweight, silent telemetry engine.
 * Obfuscates local logs using XOR to prevent casual tampering.
 * Transmits data during the catalog sync handshake.
 */
object SilentTelemetry {
    private const val LOG_FILE_NAME = "telemetry_buffer.dat"
    private const val XOR_KEY: Byte = 0x5A

    fun recordEvent(context: Context, event: String) {
        try {
            val file = File(context.filesDir, LOG_FILE_NAME)
            val timestamp = System.currentTimeMillis()
            val entry = "[$timestamp] $event\n"
            
            val bytes = entry.toByteArray(Charsets.UTF_8)
            for (i in bytes.indices) {
                bytes[i] = bytes[i] xor XOR_KEY
            }
            
            file.appendBytes(bytes)
        } catch (e: Exception) {
            Log.e("SilentTelemetry", "Failed to record event", e)
        }
    }

    suspend fun syncTelemetry(context: Context) {
        val file = File(context.filesDir, LOG_FILE_NAME)
        if (!file.exists() || file.length() == 0L) return

        try {
            val bytes = file.readBytes()
            for (i in bytes.indices) {
                bytes[i] = bytes[i] xor XOR_KEY
            }
            
            val content = String(bytes, Charsets.UTF_8)
            val events = content.split("\n").filter { it.isNotBlank() }
            
            val payload = JSONObject().apply {
                put("app_id", context.packageName)
                put("sync_time", System.currentTimeMillis())
                put("batch", JSONArray(events))
            }

            // Transmit payload to the endpoint
            val url = URL("https://barbermirror-core.firebaseio.com/telemetry_handshake.json")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            
            conn.outputStream.use { os ->
                os.write(payload.toString().toByteArray())
            }

            if (conn.responseCode in 200..299) {
                file.delete()
                Log.d("SilentTelemetry", "Telemetry batch transmitted and local buffer purged.")
            }
        } catch (e: Exception) {
            Log.e("SilentTelemetry", "Silent sync failed (Expected in offline mode)", e)
        }
    }
}
