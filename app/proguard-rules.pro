# --- HARDENING PROTOCOL: MISSION CRITICAL CORE ---

# Aggressive Obfuscation for the OTA Pipeline
-keep class com.solo.barbersmirror.OTAUpdater {
    public void downloadAndInstall(java.lang.String, java.lang.String, java.lang.String, long);
}
-keepclassmembers class com.solo.barbersmirror.OTAUpdater {
    private *** calculateSHA256(...);
    private *** installApk(...);
}

# Obfuscate the Biometric Engine
-keep class com.solo.barbersmirror.GoldenVectorEngine {
    public *** isFaceAligned(...);
    public *** analyzeFaceShape(...);
}
# Mangling the internal math to prevent algorithm extraction
-keepclassmembers class com.solo.barbersmirror.GoldenVectorEngine {
    private static native *** predictShapeNative(...);
}

# Global Shrinking & Optimization
-repackageclasses 'com.solo.bm.internal'
-allowaccessmodification
-overloadaggressively

# Maintain JNI Bridge
-keepclasseswithmembernames class * {
    native <methods>;
}

# --- THIRD-PARTY STABILITY (STRICT MINIMUM) ---
-keep class com.google.firebase.** { *; }
-keep class com.google.mediapipe.** { *; }
-keep class org.json.** { *; }
-dontwarn com.google.firebase.**
# MediaPipe Internal Protobuf Suppressions
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate
