package com.solo.barbersmirror

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import android.util.Log

@RunWith(AndroidJUnit4::class)
class AccuracyStressTest {

    @Test
    fun runFullEngineBenchmark() = runBlocking {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val assetManager = appContext.assets
        
        Log.i("STRESS_TEST", "--- DEBUG ASSET LISTING ---")
        val rootAssets = assetManager.list("") ?: emptyArray()
        rootAssets.forEach { Log.i("STRESS_TEST", "ROOT ASSET: $it") }
        
        val oblongFiles = assetManager.list("oblong_face") ?: emptyArray()
        Log.i("STRESS_TEST", "OBLONG FACE COUNT: ${oblongFiles.size}")
        oblongFiles.take(5).forEach { Log.i("STRESS_TEST", "OBLONG FILE: $it") }

        val benchmark = AccuracyBenchmark(appContext)
        
        Log.i("STRESS_TEST", "--- STARTING FULL ENGINE BENCHMARK ---")
        
        benchmark.runStressTest { stats ->
            if (!stats.isRunning) {
                // Final report
                Log.i("STRESS_TEST", "--- BENCHMARK COMPLETE ---")
                Log.i("STRESS_TEST", "TOTAL PROCESSED: ${stats.totalProcessed}")
                Log.i("STRESS_TEST", "CORRECT GUESSES: ${stats.correctGuesses}")
                Log.i("STRESS_TEST", "ACCURACY SCORE: ${"%.2f".format(stats.accuracyScore)}%")
                
                Log.i("STRESS_TEST", "--- CONFUSION MATRIX ---")
                stats.confusionMatrix.forEach { (gt, predictions) ->
                    val row = predictions.map { "${it.key}: ${it.value}" }.joinToString(" | ")
                    Log.i("STRESS_TEST", "$gt -> $row")
                }
                
                Log.i("STRESS_TEST", "--- FAILURE ANALYSIS (Sample) ---")
                stats.failures.take(20).forEach { failure ->
                    Log.i("STRESS_TEST", "FAIL: ${failure.imagePath} | GT: ${failure.groundTruth} | PREDICTED: ${failure.predicted}")
                }
            }
        }
    }
}
