package com.solo.barbersmirror

object HairstyleDatabase {

    // The Air-Gapped Asset Router
    fun getRecommendations(shape: String): List<String> {
        return when (shape.uppercase()) {
            "SQUARE" -> listOf(
                "Textured Pompadour (Softens sharp jaw)",
                "Classic Side Part (Adds asymmetrical balance)",
                "Buzz Cut (Emphasizes strong bone structure)"
            )
            "ROUND" -> listOf(
                "High Volume Quiff (Elongates the face)",
                "Faux Hawk (Draws the eye upward)",
                "Tight Fade with Spiky Top (Adds sharp angles)"
            )
            "OVAL" -> listOf(
                "Push Back / Flow (Maintains natural balance)",
                "Fringe / Crop (Avoids excessive height)",
                "Taper Fade with Short Top (Clean & structured)"
            )
            "OBLONG" -> listOf(
                "Textured French Crop (Reduces vertical length)",
                "Side Fringe (Breaks up facial length)",
                "Classic Crew Cut (Maintains width, minimal height)"
            )
            else -> listOf("AWAITING BIOMETRIC LOCK...")
        }
    }
}