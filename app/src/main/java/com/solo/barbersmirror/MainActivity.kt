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
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
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

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable

enum class ScannerState {
    TARGETING, PROCESSING, LOCKED
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
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }

    val hudColor = when {
        currentState == ScannerState.TARGETING && !isLiveAligned -> Color.Red
        currentState == ScannerState.TARGETING && isLiveAligned -> Color(0xFF00FFCC)
        targetShape == "RE-ALIGN FACE" || targetShape == "NO_TARGET" || targetShape == "SYS_ERR" || targetShape == "MATH_ERR" -> Color.Red
        else -> Color(0xFF00FFCC)
    }

    val imageCapture = remember { ImageCapture.Builder().build() }

    val faceLandmarker = remember {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE)
                .setNumFaces(1)
                .build()
            FaceLandmarker.createFromOptions(context, options)
        } catch (_: Exception) {
            null
        }
    }

    fun runAnalysis(bitmap: Bitmap) {
        if (faceLandmarker == null) {
            targetShape = "SYS_ERR"
            rawTelemetry = "MESH ENGINE OFFLINE"
            capturedImage = bitmap
            currentState = ScannerState.LOCKED
            return
        }

        try {
            val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val mpImage = BitmapImageBuilder(argbBitmap).build()
            val result = faceLandmarker.detect(mpImage)

            if (result.faceLandmarks().isEmpty()) {
                targetShape = "NO_TARGET"
                rawTelemetry = "NO FACE DETECTED IN FEED"
                capturedImage = argbBitmap
                currentState = ScannerState.LOCKED
                return
            }

            val mesh = result.faceLandmarks()[0]

            if (GoldenVectorEngine.isFaceAligned(mesh)) {
                val result3D = GoldenVectorEngine.analyzeFaceShape(mesh)
                targetShape = result3D.shape
                rawTelemetry = "BIOMETRIC LOCK SECURED | H/W: ${"%.2f".format(result3D.lRatio)} | J/W: ${"%.2f".format(result3D.fRatio)}"
                capturedImage = argbBitmap
                currentState = ScannerState.LOCKED
            } else {
                targetShape = "RE-ALIGN FACE"
                rawTelemetry = "ALIGN FACE STRAIGHT INTO CAMERA"
                capturedImage = argbBitmap
                currentState = ScannerState.LOCKED
            }

        } catch (_: Exception) {
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
                    val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    runAnalysis(argbBitmap)
                } catch (_: Exception) {
                }
            }
        }
    )

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        if (hasPermission) {
            if (currentState == ScannerState.TARGETING || currentState == ScannerState.PROCESSING) {
                CameraPreview(
                    imageCapture = imageCapture,
                    faceLandmarker = faceLandmarker,
                    lensFacing = lensFacing,
                    onAlignmentChange = { aligned ->
                        isLiveAligned = aligned
                    }
                )
            } else if (currentState == ScannerState.LOCKED && capturedImage != null) {
                Image(
                    bitmap = capturedImage!!.asImageBitmap(),
                    contentDescription = "Captured Scan",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                // Darken the background to make the catalog pop
                Box(modifier = Modifier.fillMaxSize().background(Color(0x88000000)))
            }

            if (currentState == ScannerState.TARGETING || currentState == ScannerState.PROCESSING || targetShape == "RE-ALIGN FACE") {
                Box(
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(bottom = 50.dp)
                        .width(220.dp)
                        .height(300.dp)
                        .border(2.dp, hudColor.copy(alpha = 0.8f), RoundedCornerShape(150.dp))
                )

                if (currentState != ScannerState.LOCKED) {
                    Text(
                        text = "ALIGN FACE WITHIN BRACKETS",
                        color = Color(0xFF00FFCC),
                        modifier = Modifier.align(Alignment.TopCenter).padding(top = 80.dp),
                        letterSpacing = 2.sp
                    )
                }
            }

            if (currentState == ScannerState.TARGETING) {
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(bottom = 60.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = { photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                    ) {
                        Text("GALLERY", color = Color.White)
                    }

                    Button(
                        onClick = {
                            if (isLiveAligned) {
                                currentState = ScannerState.PROCESSING
                                imageCapture.takePicture(
                                    ContextCompat.getMainExecutor(context),
                                    object : ImageCapture.OnImageCapturedCallback() {
                                        override fun onCaptureSuccess(image: ImageProxy) {
                                            val buffer: ByteBuffer = image.planes[0].buffer
                                            val bytes = ByteArray(buffer.remaining())
                                            buffer.get(bytes)
                                            val rawBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, null)

                                            val matrix = Matrix().apply {
                                                postRotate(image.imageInfo.rotationDegrees.toFloat())
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
                                            currentState = ScannerState.TARGETING
                                        }
                                    }
                                )
                            }
                        },
                        modifier = Modifier.size(80.dp),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isLiveAligned) Color.White else Color.DarkGray
                        )
                    ) {}

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

            // --- THE 2D CATALOG & HERO UI ---
            if (currentState == ScannerState.LOCKED && targetShape != "ERROR" && targetShape != "SYS_ERR" && targetShape != "MATH_ERR" && targetShape != "NO_TARGET" && targetShape != "RE-ALIGN FACE") {

                val shapeLower = targetShape.lowercase()
                val hairline = "solid_hairline" // Can be made dynamic later
                val basePath = "hair_models/${shapeLower}_face/$hairline"

                // Scan directory for available _front.png styles
                val availableStyles = remember(targetShape) {
                    try {
                        context.assets.list(basePath)?.filter { it.endsWith("_front.png") } ?: emptyList()
                    } catch (_: Exception) {
                        emptyList()
                    }
                }

                // UI States
                var focusedStyle by remember { mutableStateOf<String?>(null) }
                var show360 by remember { mutableStateOf(false) }
                var scale by remember { mutableFloatStateOf(1f) }
                var offset by remember { mutableStateOf(Offset.Zero) }

                if (focusedStyle == null) {
                    // LAYER 1A: THE ASSET GRID
                    LazyVerticalGrid(
                        columns = GridCells.Fixed(2),
                        contentPadding = PaddingValues(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(top = 60.dp, bottom = 200.dp) // Space for HUD
                    ) {
                        items(availableStyles) { styleFileName ->
                            val path = "$basePath/$styleFileName"
                            val bmp = remember(path) {
                                try { BitmapFactory.decodeStream(context.assets.open(path)) } catch (_: Exception) { null }
                            }
                            if (bmp != null) {
                                Image(
                                    bitmap = bmp.asImageBitmap(),
                                    contentDescription = "Style Option",
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier
                                        .aspectRatio(1f) // Keep images square in the grid
                                        .clip(RoundedCornerShape(12.dp))
                                        .border(2.dp, Color(0xFF00FFCC).copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                                        .clickable {
                                            focusedStyle = styleFileName
                                            scale = 1f
                                            offset = Offset.Zero
                                        }
                                )
                            }
                        }
                    }
                } else {
                    // LAYER 1B: THE HERO FOCUS VIEW
                    val transformableState = rememberTransformableState { zoomChange, offsetChange, _ ->
                        scale = (scale * zoomChange).coerceIn(1f, 4f)
                        offset += offsetChange
                    }

                    val assetSuffix = if (show360) "_360.png" else "_front.png"
                    val targetFile = focusedStyle!!.replace("_front.png", assetSuffix)
                    val assetPath = "$basePath/$targetFile"

                    val lookbookBitmap = remember(assetPath) {
                        try { BitmapFactory.decodeStream(context.assets.open(assetPath)) } catch (_: Exception) { null }
                    }

                    if (lookbookBitmap != null) {
                        Image(
                            bitmap = lookbookBitmap.asImageBitmap(),
                            contentDescription = "Focused Style",
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(bottom = 180.dp)
                                .graphicsLayer(
                                    scaleX = scale,
                                    scaleY = scale,
                                    translationX = offset.x,
                                    translationY = offset.y
                                )
                                .transformable(state = transformableState)
                        )
                    }

                    // Top Action Bar for Hero View
                    Row(
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .fillMaxWidth()
                            .padding(top = 40.dp, start = 20.dp, end = 20.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Button(
                            onClick = {
                                focusedStyle = null
                                show360 = false
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                        ) {
                            Text("BACK TO GRID", color = Color.White, fontWeight = FontWeight.Bold)
                        }

                        Button(
                            onClick = {
                                show360 = !show360
                                scale = 1f
                                offset = Offset.Zero
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00FFCC))
                        ) {
                            Text(
                                text = if (show360) "VIEW FRONT" else "VIEW 360°",
                                color = Color.Black,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }

            // HUD OVERLAY
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
                        fontWeight = FontWeight.Black,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(rawTelemetry, color = Color.DarkGray, fontSize = 12.sp, modifier = Modifier.padding(bottom = 24.dp))

                    Button(
                        onClick = {
                            currentState = ScannerState.TARGETING
                            capturedImage = null
                            targetShape = ""
                            rawTelemetry = ""
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
                    ) {
                        Text("REBOOT SCANNER", color = Color.White, letterSpacing = 2.sp)
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

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val mainExecutor = ContextCompat.getMainExecutor(ctx)

            cameraProviderFuture.addListener({
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
                                } catch (_: Exception) {
                                }
                            }
                            imageProxy.close()
                        }
                    }

                val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, cameraSelector, preview, imageCapture, imageAnalyzer
                    )
                } catch (_: Exception) {
                }
            }, mainExecutor)

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}