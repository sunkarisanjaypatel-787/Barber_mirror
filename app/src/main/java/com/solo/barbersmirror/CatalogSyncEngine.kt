package com.solo.barbersmirror

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.URL

/**
 * Handles the transactional synchronization of the hairstyle catalog.
 * Saves assets to specific filenames for direct manual loading.
 */
suspend fun synchronizeCatalog(
    context: Context, 
    latestVersion: Int, 
    latestManifest: JSONObject,
    firebaseBaseUrl: String
) = withContext(Dispatchers.IO) {
    
    val prefs = context.getSharedPreferences("barber_mirror_sys", Context.MODE_PRIVATE)
    val localVersion = prefs.getInt("local_manifest_version", -1)

    // 1. VERSION CHECK: If matched, abort sync. System is autonomous.
    if (localVersion == latestVersion) {
        return@withContext 
    }

    val cacheDir = File(context.filesDir, "catalog_matrix_cache")
    if (!cacheDir.exists()) {
        cacheDir.mkdirs()
    }

    val targets = parseManifestForTargets(latestManifest, firebaseBaseUrl)

    // 2. THE PURGE: Annihilate the legacy cache
    cacheDir.listFiles()?.forEach { it.delete() }

    // 3. THE PREFETCH: Manual download to predictable filenames
    var successfulFetches = 0

    for (target in targets) {
        try {
            val destination = File(cacheDir, target.fileName)
            URL(target.url).openStream().use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            successfulFetches++
        } catch (_: Exception) {
            // Log or handle individual failures
        }
    }

    // 4. TRANSACTION VERIFICATION
    if (successfulFetches == targets.size && targets.isNotEmpty()) {
        // Lock in the new state. The system is now fully synced and offline-ready.
        prefs.edit().putInt("local_manifest_version", latestVersion).apply()
        
        // PIGGYBACK TELEMETRY: Sync logs silently during manifest handshake
        SilentTelemetry.syncTelemetry(context)
    }
}

private data class SyncTarget(val url: String, val fileName: String)

/**
 * Parses the manifest JSON to extract URL and filename pairs.
 */
private fun parseManifestForTargets(manifest: JSONObject, baseUrl: String): List<SyncTarget> {
    val targets = mutableListOf<SyncTarget>()
    val shapes = manifest.keys()
    while (shapes.hasNext()) {
        val shape = shapes.next()
        val styles = manifest.optJSONArray(shape) ?: continue
        for (i in 0 until styles.length()) {
            val styleId = styles.getInt(i)
            
            // Front View
            targets.add(SyncTarget(
                url = "${baseUrl}hair_models%2F${shape}_face%2Ffront%2F${shape}_solid_front_${styleId}.webp?alt=media",
                fileName = "${shape}_solid_front_${styleId}.webp"
            ))
            
            // 360 View
            targets.add(SyncTarget(
                url = "${baseUrl}hair_models%2F${shape}_face%2F360%2F${shape}_solid_360_${styleId}.webp?alt=media",
                fileName = "${shape}_solid_360_${styleId}.webp"
            ))
        }
    }
    return targets
}
