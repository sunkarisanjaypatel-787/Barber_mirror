package com.solo.barbersmirror

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import androidx.camera.core.ImageAnalysis

// --- THE GEOMETRIC BRIDGE IMPORTS ---
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker

import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput

// THE STATE MACHINE
enum class ScannerState {
    TARGETING, PROCESSING, LOCKED,
}

class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        setContent { CyberpunkScanner(hasPermission = isGranted) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val hasCamera = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (hasCamera) {
            setContent { CyberpunkScanner(hasPermission = true) }
        } else {
            setContent { CyberpunkScanner(hasPermission = false) }
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
}

@Composable
fun CyberpunkScanner(hasPermission: Boolean) {
    val context = LocalContext.current
    var currentState by remember { mutableStateOf(ScannerState.TARGETING) }
    var targetShape by remember { mutableStateOf("") }
    var rawTelemetry by remember { mutableStateOf("") }
    var capturedImage by remember { mutableStateOf<Bitmap?>(null) }
    var isLiveAligned by remember { mutableStateOf(false) }

    // 1. INJECT LENS STATE (Defaults to Front)
    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_FRONT) }

    // --- THE REACTIVE HUD LOGIC ---
    val hudColor = when {
        currentState == ScannerState.TARGETING && !isLiveAligned -> Color.Red
        currentState == ScannerState.TARGETING && isLiveAligned -> Color(0xFF00FFCC) // Turns Teal when aligned
        targetShape == "RE-ALIGN FACE" || targetShape == "NO_TARGET" || targetShape == "SYS_ERR" || targetShape == "MATH_ERR" -> Color.Red
        else -> Color(0xFF00FFCC)
    }

    // The Hardware Shutter
    val imageCapture = remember { ImageCapture.Builder().build() }

    // --- PHASE 1: INITIALIZE NATIVE C++ ENGINE ---
    val faceLandmarker = remember {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()
            // Configure for static image mode, searching for a single face
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE)
                .setNumFaces(1)
                .build()
            FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            println("MESH ENGINE FAILURE: ${e.message}")
            null
        }
    }

    // --- PHASE 2: THE GEOMETRIC CALCULUS BLOCK ---
    fun runAnalysis(bitmap: Bitmap) {
        if (faceLandmarker == null) {
            targetShape = "SYS_ERR"
            rawTelemetry = "MESH ENGINE OFFLINE"
            capturedImage = bitmap
            currentState = ScannerState.LOCKED
            return
        }

        try {
            // MediaPipe requires the bitmap to be strictly ARGB_8888
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

            // 1. Data Conversion
            val mpImage = BitmapImageBuilder(argbBitmap).build()

            // 2. Execute Mesh Projection
            val result = faceLandmarker.detect(mpImage)

            // Guard: Face not detected in frame
            if (result.faceLandmarks().isEmpty()) {
                targetShape = "NO_TARGET"
                rawTelemetry = "NO FACE DETECTED IN FEED"
                capturedImage = argbBitmap
                currentState = ScannerState.LOCKED
                return
            }

            // 3. Extract the 3D Coordinate Mesh
            val mesh = result.faceLandmarks()[0]

            // --- PROTOCOL GAMMA: THE KYC LOCK ---
            if (GoldenVectorEngine.isFaceAligned(mesh)) {
                // 4. Golden Vector Classification (3D Euclidean Logic)
                val result3D = GoldenVectorEngine.analyzeFaceShape(mesh)

                // 5. State Machine Handoff
                targetShape = result3D.shape
                // 3D telemetry for accurate debugging
                rawTelemetry = "BIOMETRIC LOCK SECURED | 3D L: ${"%.2f".format(result3D.lRatio)} | F: ${"%.2f".format(result3D.fRatio)} | G: ${"%.2f".format(result3D.gRatio)} | M: ${"%.2f".format(result3D.mRatio)}"
                capturedImage = argbBitmap // Full uncropped frame
                currentState = ScannerState.LOCKED
            } else {
                // BLOCK THE MATH - User is tilted
                targetShape = "RE-ALIGN FACE"
                rawTelemetry = "ALIGN FACE STRAIGHT INTO CAMERA"
                capturedImage = argbBitmap
                currentState = ScannerState.LOCKED
            }

        } catch (e: Exception) {
            println("GEOMETRIC FRACTURE: ${e.message}")
            targetShape = "MATH_ERR"
            rawTelemetry = "PIPELINE COLLAPSE: CHECK LOGS"
            capturedImage = bitmap
            currentState = ScannerState.LOCKED
        }
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            if (uri != null) {
                try {
                    val bitmap = if (Build.VERSION.SDK_INT >= 28) {
                        val source = ImageDecoder.createSource(context.contentResolver, uri)
                        ImageDecoder.decodeBitmap(source)
                    } else {
                        @Suppress("DEPRECATION")
                        MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                    }
                    currentState = ScannerState.PROCESSING
                    
                    // Critical Fix: Force ARGB_8888 for Gallery Images
                    val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    runAnalysis(argbBitmap)
                } catch (e: Exception) {
                    println("GALLERY UPLINK FAILED: ${e.message}")
                }
            }
        }
    )

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) { if (hasPermission) {

            // --- LAYER 1: THE VISUAL FEED ---
            if (currentState == ScannerState.TARGETING || currentState == ScannerState.PROCESSING) {
                CameraPreview(
                    imageCapture = imageCapture,
                    faceLandmarker = faceLandmarker,
                    lensFacing = lensFacing,
                    onAlignmentChange = { aligned ->
                        isLiveAligned = aligned // Updates UI instantly
                    }
                )
            } else if (currentState == ScannerState.LOCKED && capturedImage != null) {
                // Show the frozen, captured image
                Image(
                    bitmap = capturedImage!!.asImageBitmap(),
                    contentDescription = "Captured Scan",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = androidx.compose.ui.layout.ContentScale.Crop
                )
                // Darken it slightly so the UI pops
                Box(modifier = Modifier.fillMaxSize().background(Color(0x66000000)))
            }

            // --- LAYER 2: THE HUD OVERLAY (Visible during targeting or failure) ---
            if (currentState == ScannerState.TARGETING || currentState == ScannerState.PROCESSING || targetShape == "RE-ALIGN FACE") {
                // The Targeting Guideline (Reactive color)
                Box(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(bottom = 50.dp)
                        .width(220.dp)
                        .height(300.dp)
                        .border(2.dp, hudColor.copy(alpha = 0.8f), RoundedCornerShape(150.dp)) // Glowing Oval
                )

                if (currentState != ScannerState.LOCKED) {
                    Text(
                        text = "ALIGN FACE WITHIN BRACKETS",
                        color = Color(0xFF00FFCC),
                        // PRODUCTION PATCH: Severed the long-press backdoor pointer input
                        modifier = Modifier.align(Alignment.TopCenter).padding(top = 80.dp),
                        letterSpacing = 2.sp
                    )
                }
            }

            // --- LAYER 3: THE UI CONTROLS ---

            // SHUTTER CONTROLS (Only visible when targeting)
            if (currentState == ScannerState.TARGETING) {
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(bottom = 60.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Gallery Button
                    Button(
                        onClick = { photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                    ) {
                        Text("GALLERY", color = Color.White)
                    }

                    // The Main Shutter Button
                    Button(
                        onClick = {
                            if (isLiveAligned) { // <-- THE HARDWARE LOCK
                                currentState = ScannerState.PROCESSING
                                imageCapture.takePicture(
                                    ContextCompat.getMainExecutor(context),
                                    object : ImageCapture.OnImageCapturedCallback() {
                                        override fun onCaptureSuccess(image: ImageProxy) {
                                            // Convert the raw sensor data into a correct mirror-image Bitmap
                                            val buffer: ByteBuffer = image.planes[0].buffer
                                            val bytes = ByteArray(buffer.remaining())
                                            buffer.get(bytes)
                                            val rawBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, null)

                                            val matrix = Matrix().apply {
                                                postRotate(image.imageInfo.rotationDegrees.toFloat())
                                                
                                                // 2. CONDITIONAL MIRRORING (Only mirror the front camera)
                                                if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                                                    postScale(-1f, 1f, rawBitmap.width / 2f, rawBitmap.height / 2f)
                                                }
                                            }
                                            val rotatedBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                                            val argbBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)

                                            image.close()
                                            runAnalysis(argbBitmap)
                                        }

                                        override fun onError(exc: ImageCaptureException) {
                                            println("SHUTTER FAILURE: ${exc.message}")
                                            currentState = ScannerState.TARGETING
                                        }
                                    }
                                )
                            }
                        },
                        modifier = Modifier.size(80.dp),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            // Button glows white when ready, dark grey when locked
                            containerColor = if (isLiveAligned) Color.White else Color.DarkGray
                        )
                    ) {
                        // Invisible text, it's just a white circle like a real camera
                    }

                    // INJECT THE CAMERA FLIP BUTTON
                    Button(
                        onClick = { 
                            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                                CameraSelector.LENS_FACING_BACK
                            } else {
                                CameraSelector.LENS_FACING_FRONT
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                    ) {
                        Text("FLIP LENS", color = Color.White)
                    }
                }
            }

            // PROCESSING SCREEN
            if (currentState == ScannerState.PROCESSING) {
                Box(modifier = Modifier.fillMaxSize().background(Color(0x99000000))) {
                    Text(
                        text = "EXTRACTING BIOMETRICS...",
                        color = Color(0xFF00FFCC),
                        fontSize = 18.sp,
                        letterSpacing = 3.sp,
                        modifier = Modifier.align(Alignment.Center)
                    )
                }
            }

            // THE DATA PANEL (Only visible when Locked)
            if (currentState == ScannerState.LOCKED) {
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .background(Color(0xEE121212), shape = RoundedCornerShape(topStart = 30.dp, topEnd = 30.dp))
                        .padding(24.dp)
                ) {
                    Text("BIOMETRIC LOCK:", color = Color.Gray, fontSize = 12.sp, letterSpacing = 1.5.sp)
                    Text(
                        targetShape.uppercase(),
                        color = if (targetShape == "ERROR" || targetShape == "SYS_ERR" || targetShape == "MATH_ERR" || targetShape == "NO_TARGET" || targetShape == "RE-ALIGN FACE") Color.Red else Color(0xFF00FFCC),
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(rawTelemetry, color = Color.DarkGray, fontSize = 12.sp, modifier = Modifier.padding(bottom = 24.dp))

                    if (targetShape != "ERROR" && targetShape != "SYS_ERR" && targetShape != "MATH_ERR" && targetShape != "NO_TARGET" && targetShape != "RE-ALIGN FACE") {
                        Text("OPTIMAL HAIRSTYLES:", color = Color.White, fontSize = 14.sp, modifier = Modifier.padding(bottom = 8.dp))
                        
                        // THE MATRIX INTEGRATION
                        val recommendations = HairstyleDatabase.getRecommendations(targetShape)
                        recommendations.forEach { style ->
                            Text("> $style", color = Color(0xFF00FFCC), fontSize = 12.sp, modifier = Modifier.padding(bottom = 4.dp))
                        }
                    }

                    Spacer(modifier = Modifier.height(30.dp))

                    Button(
                        onClick = {
                            currentState = ScannerState.TARGETING
                            capturedImage = null
                            targetShape = "" // <-- INJECT THIS: Resets HUD to Teal
                            rawTelemetry = "" // <-- INJECT THIS: Clears old data
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                    ) {
                        Text("RETAKE SCAN", color = Color.White, letterSpacing = 2.sp)
                    }
                }
            }
        } else {
            Text("HARDWARE OFFLINE", color = Color.Red, modifier = Modifier.align(Alignment.Center))
        }
    }
}

@Composable
fun CameraPreview(
    imageCapture: ImageCapture,
    faceLandmarker: FaceLandmarker?,
    lensFacing: Int,
    onAlignmentChange: (Boolean) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }
    
    // 1. Decouple the View initialization from the Camera logic
    val previewView = remember { PreviewView(context) }

    // 2. THE HARDWARE REBOOT TRIGGER
    // This block automatically re-executes every time 'lensFacing' changes
    LaunchedEffect(lensFacing) {
        val cameraProvider = cameraProviderFuture.get()
        
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                    if (faceLandmarker != null) {
                        try {
                            val bitmap = imageProxy.toBitmap()
                            val matrix = Matrix().apply {
                                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                                
                                // CONDITIONAL MIRRORING (Now safely updates with state)
                                if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                                    postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
                                }
                            }
                            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                            val argbBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                            val mpImage = BitmapImageBuilder(argbBitmap).build()

                            val result = faceLandmarker.detect(mpImage)
                            if (result.faceLandmarks().isNotEmpty()) {
                                val isAligned = GoldenVectorEngine.isFaceAligned(result.faceLandmarks()[0])
                                onAlignmentChange(isAligned)
                            } else {
                                onAlignmentChange(false)
                            }
                        } catch (e: Exception) {
                            // Drop frame silently if engine lags
                        }
                    }
                    imageProxy.close() // CRITICAL: Free the buffer
                }
            }

        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        try {
            // 3. Purge the old camera feed and bind the new one
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner, cameraSelector, preview, imageCapture, imageAnalyzer
            )
        } catch (e: Exception) {
            println("OPTICAL BRIDGE FAILURE: ${e.message}")
        }
    }

    // 4. A clean, passive AndroidView that just renders whatever LaunchedEffect feeds it
    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}
