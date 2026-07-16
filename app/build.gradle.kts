
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    //id("com.android.application")
    // Add the Google services Gradle plugin
    id("com.google.gms.google-services")
}
// Place this at the top level of app/build.gradle.kts to prevent D8 duplicate class crashes
configurations.all {
    exclude(group = "org.tensorflow")
}
android {
    namespace = "com.solo.barbersmirror"
    compileSdk=35

    defaultConfig {
        applicationId = "com.solo.barbersmirror"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = true // ENABLE R8
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    sourceSets {
        named("main") {
            assets.srcDirs("src/main/assets", "src/main/others")
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
    // CameraX (Zero-Latency Optical Pipeline)
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // MediaPipe Tasks Vision (Hardware-Accelerated ML Engine)
    implementation("com.google.mediapipe:tasks-vision:0.10.10")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")

    // LiteRT (The 16KB-aligned successor to TensorFlow Lite)
    implementation("com.google.ai.edge.litert:litert:1.4.0")
    implementation("com.google.ai.edge.litert:litert-support:1.4.0")

    // PROTOCOL V2.0: The 3D Rendering Engine (SceneView / Filament)
    implementation("io.github.sceneview:sceneview:4.11.2")

    // Coil Image Loader (With Compose support)
    implementation("io.coil-kt:coil-compose:2.5.0")
    // Add the Google services Gradle plugin
    implementation(platform("com.google.firebase:firebase-bom:34.16.0"))
    implementation("com.google.firebase:firebase-config")
    implementation("com.google.firebase:firebase-analytics")
    // Forces the compiler to use a modern Fragment version, resolving the ActivityResult Lint crash
    implementation("androidx.fragment:fragment-ktx:1.6.2")
}
