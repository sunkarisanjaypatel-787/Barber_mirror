package com.solo.barbersmirror

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File
import java.security.MessageDigest

class OTAUpdater(private val context: Context) {

    fun downloadAndInstall(apkUrl: String, version: String, targetVersionCode: Long, expectedSha256: String) {
        val currentVersionCode = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode
        } else {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0).versionCode.toLong()
        }

        // ANTI-DOWNGRADE PROTECTION: Guard against rollback attacks
        if (targetVersionCode <= currentVersionCode) {
            Log.w("OTAUpdater", "REJECTED: Update version code $targetVersionCode is not greater than current $currentVersionCode")
            return
        }

        val destination = File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "BarbersMirror_v$version.apk")
        
        // Purge old payload if it exists to ensure a clean write
        if (destination.exists()) destination.delete()

        val request = DownloadManager.Request(Uri.parse(apkUrl))
            .setTitle("Barber's Mirror System Update")
            .setDescription("Downloading v$version payload...")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setDestinationUri(Uri.fromFile(destination))
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)

        val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val downloadId = manager.enqueue(request)

        // Listen for the exact moment the payload hits the disk
        val onComplete = object : BroadcastReceiver() {
            override fun onReceive(ctx: Context, intent: Intent) {
                val id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1)
                if (id == downloadId) {
                    context.unregisterReceiver(this)
                    
                    // Verify cryptographic integrity before execution
                    val fileHash = calculateSHA256(destination)
                    if (fileHash.equals(expectedSha256, ignoreCase = true)) {
                        installApk(destination)
                    } else {
                        Log.e("OTAUpdater", "CRITICAL: SHA-256 mismatch. Expected $expectedSha256 but got $fileHash.")
                        if (destination.exists()) destination.delete() // Purge compromised binary
                    }
                }
            }
        }
        
        // Context.RECEIVER_EXPORTED is mandatory for Android 14+ to receive DownloadManager system broadcasts
        val receiverFlags = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            Context.RECEIVER_EXPORTED
        } else {
            0
        }

        context.registerReceiver(
            onComplete, 
            IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE), 
            receiverFlags
        )
    }

    private fun calculateSHA256(file: File): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { fis ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (fis.read(buffer).also { bytesRead = it } != -1) {
                    digest.update(buffer, 0, bytesRead)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            Log.e("OTAUpdater", "SHA-256 calculation failed", e)
            ""
        }
    }

    private fun installApk(apkFile: File) {
        // Generates a secure URI that the Android OS Installer is allowed to read.
        // Changed to .fileprovider to match standard manifest naming conventions.
        val uri = FileProvider.getUriForFile(
            context, 
            "${context.packageName}.fileprovider", 
            apkFile
        )
        
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            // FLAG_GRANT_READ_URI_PERMISSION is the master key to bypass the Sandbox
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION
        }
        context.startActivity(intent)
    }
}