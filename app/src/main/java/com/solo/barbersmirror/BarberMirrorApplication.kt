package com.solo.barbersmirror

import android.app.Application
import coil.Coil
import coil.ImageLoader
import coil.disk.DiskCache

class BarberMirrorApplication : Application() {
    override fun onCreate() {
        super.onCreate()

        val imageLoader = ImageLoader.Builder(this)
            .diskCache {
                DiskCache.Builder()
                    .directory(this.filesDir.resolve("catalog_matrix_cache"))
                    .maxSizeBytes(512L * 1024 * 1024) 
                    .build()
            }
            .respectCacheHeaders(false) // Force trust in local disk
            .build()

        Coil.setImageLoader(imageLoader)
    }
}
