package com.solo.barbersmirror

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sqrt

data class ClassificationResult(
    val shape: String,
    val lRatio: Float,
    val fRatio: Float,
    val gRatio: Float,
    val mRatio: Float
)

object GoldenVectorEngine {

    val SHAPES = arrayOf("oblong", "oval", "round", "square", "triangle")

    // --- INJECTED BRIDGE LOGIC ---
    private fun calcDist(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        val dx = p1.x() - p2.x()
        val dy = p1.y() - p2.y()
        val dz = p1.z() - p2.z()
        return sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2))
    }

    private fun calcDist2D(p1: NormalizedLandmark, p2: NormalizedLandmark): Float {
        val dx = p1.x() - p2.x()
        val dy = p1.y() - p2.y()
        return sqrt(dx.pow(2) + dy.pow(2))
    }

    fun isFaceAligned(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size < 468) return false
        val noseToLeftCheek = calcDist2D(landmarks[1], landmarks[234])
        val noseToRightCheek = calcDist2D(landmarks[1], landmarks[454])
        val yawRatio = if (noseToRightCheek > 0) noseToLeftCheek / noseToRightCheek else 0f

        val noseToChin = calcDist2D(landmarks[1], landmarks[152])
        val noseToForehead = calcDist2D(landmarks[1], landmarks[10])
        val pitchRatio = if (noseToForehead > 0) noseToChin / noseToForehead else 0f

        val cheekDepthDiff = abs(landmarks[234].z() - landmarks[454].z())
        val isZLocked = cheekDepthDiff < 0.04f

        return (yawRatio in 0.85f..1.15f) && (pitchRatio in 0.80f..1.20f) && isZLocked
    }

    fun analyzeFaceShape(landmarks: List<NormalizedLandmark>): ClassificationResult {
        if (landmarks.size < 468) return ClassificationResult("UNDETERMINED", 0f, 0f, 0f, 0f)

        // Extract the 4D matrix directly from the 3D mesh
        val faceHeight = calcDist(landmarks[10], landmarks[152])
        val faceWidth = calcDist(landmarks[234], landmarks[454])
        val jawWidth = calcDist(landmarks[132], landmarks[361])
        val foreheadWidth = calcDist(landmarks[54], landmarks[284])

        val h_w = if (faceWidth > 0) faceHeight / faceWidth else 0f
        val j_w = if (faceWidth > 0) jawWidth / faceWidth else 0f
        val f_j = if (jawWidth > 0) foreheadWidth / jawWidth else 0f
        val c_j = if (jawWidth > 0) faceWidth / jawWidth else 0f

        // Feed the ratios into your ML decision trees
        val shape = predictShape(h_w, j_w, f_j, c_j)

        return ClassificationResult(shape.uppercase(), h_w, j_w, f_j, c_j)
    }

    // --- YOUR EXISTING ML LOGIC REMAINS BELOW ---
    fun predictShape(h_w: Float, j_w: Float, f_j: Float, c_j: Float): String {
        val votes = IntArray(5)

        // Delegate to smaller chunked methods to bypass the JVM 64KB method limit
        evaluateTrees0to9(h_w, j_w, f_j, c_j, votes)
        evaluateTrees10to19(h_w, j_w, f_j, c_j, votes)
        evaluateTrees20to29(h_w, j_w, f_j, c_j, votes)
        evaluateTrees30to39(h_w, j_w, f_j, c_j, votes)
        evaluateTrees40to49(h_w, j_w, f_j, c_j, votes)
        var maxVotes = 0
        var bestShapeIdx = 0
        for (i in votes.indices) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i]
                bestShapeIdx = i
            }
        }
        return SHAPES[bestShapeIdx]
    }

    private fun evaluateTrees0to9(h_w: Float, j_w: Float, f_j: Float, c_j: Float, votes: IntArray) {
        // Move Tree 0 through Tree 9 here
        // Tree 0
        if (j_w <= 0.97898f) {
            if (h_w <= 1.15485f) {
                if (f_j <= 0.89400f) {
                    if (c_j <= 1.05113f) {
                        if (f_j <= 0.88600f) {
                            if (f_j <= 0.86087f) {
                                if (h_w <= 1.14290f) {
                                    if (f_j <= 0.84567f) {
                                        if (c_j <= 1.03144f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.15149f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (f_j <= 0.88611f) {
                            votes[4]++
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (f_j <= 0.91169f) {
                        votes[2]++
                    } else {
                        if (f_j <= 0.91448f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.94293f) {
                                if (j_w <= 0.93977f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (c_j <= 1.03574f) {
                        if (h_w <= 1.19563f) {
                            if (j_w <= 0.97032f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.03041f) {
                                    if (j_w <= 0.97076f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.02920f) {
                                            if (h_w <= 1.17964f) {
                                                if (j_w <= 0.97272f) {
                                                    if (j_w <= 0.97248f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.84100f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.02459f) {
                                                        votes[0]++
                                                    } else {
                                                        if (f_j <= 0.88206f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (f_j <= 0.84896f) {
                                if (j_w <= 0.97047f) {
                                    if (f_j <= 0.83129f) {
                                        if (h_w <= 1.21710f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.82832f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.22440f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.84147f) {
                                        if (c_j <= 1.02492f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.20899f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.22367f) {
                                    if (f_j <= 0.88046f) {
                                        if (c_j <= 1.03274f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03391f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.03499f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.97182f) {
                                        if (j_w <= 0.96639f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.87488f) {
                                                if (f_j <= 0.86915f) {
                                                    if (j_w <= 0.96957f) {
                                                        votes[0]++
                                                    } else {
                                                        if (c_j <= 1.03124f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.97316f) {
                                            votes[1]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.94847f) {
                            if (f_j <= 0.90286f) {
                                if (c_j <= 1.05541f) {
                                    if (f_j <= 0.89544f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (c_j <= 1.05674f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.94010f) {
                                        if (h_w <= 1.27397f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.24453f) {
                                            if (f_j <= 0.91864f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.03632f) {
                                if (j_w <= 0.96545f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (j_w <= 0.95639f) {
                                    if (f_j <= 0.89923f) {
                                        if (j_w <= 0.95219f) {
                                            if (c_j <= 1.05119f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.05374f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (c_j <= 1.04877f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04204f) {
                                        if (j_w <= 0.95999f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.96308f) {
                                                if (c_j <= 1.03926f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.04072f) {
                                                        if (f_j <= 0.89811f) {
                                                            if (c_j <= 1.03974f) {
                                                                votes[4]++
                                                            } else {
                                                                if (j_w <= 0.96128f) {
                                                                    votes[4]++
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        if (c_j <= 1.04093f) {
                                                            votes[0]++
                                                        } else {
                                                            if (j_w <= 0.96038f) {
                                                                votes[0]++
                                                            } else {
                                                                if (c_j <= 1.04113f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.96436f) {
                                                    if (h_w <= 1.17554f) {
                                                        votes[0]++
                                                    } else {
                                                        if (j_w <= 0.96383f) {
                                                            votes[1]++
                                                        } else {
                                                            if (h_w <= 1.24235f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.20065f) {
                                            if (j_w <= 0.95711f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (f_j <= 0.86785f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        } else {
            if (h_w <= 1.15972f) {
                if (f_j <= 0.87120f) {
                    votes[2]++
                } else {
                    votes[3]++
                }
            } else {
                if (j_w <= 0.99157f) {
                    if (h_w <= 1.19092f) {
                        votes[3]++
                    } else {
                        if (c_j <= 1.01536f) {
                            if (f_j <= 0.79735f) {
                                votes[3]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (j_w <= 0.98044f) {
                                votes[3]++
                            } else {
                                if (j_w <= 0.98216f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.01646f) {
                                        if (c_j <= 1.01641f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.83037f) {
                        votes[0]++
                    } else {
                        votes[3]++
                    }
                }
            }
        }
        // Tree 1
        if (c_j <= 1.01639f) {
            if (f_j <= 0.79505f) {
                votes[3]++
            } else {
                if (h_w <= 1.14488f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.01268f) {
                        votes[4]++
                    } else {
                        if (f_j <= 0.83198f) {
                            votes[3]++
                        } else {
                            votes[4]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15753f) {
                if (j_w <= 0.96706f) {
                    if (h_w <= 1.12217f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.95729f) {
                            if (j_w <= 0.94638f) {
                                if (h_w <= 1.15304f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (h_w <= 1.13065f) {
                                    votes[0]++
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (h_w <= 1.13840f) {
                                if (f_j <= 0.90440f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (c_j <= 1.04341f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.12024f) {
                        if (c_j <= 1.02165f) {
                            votes[2]++
                        } else {
                            if (j_w <= 0.97690f) {
                                votes[2]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.15731f) {
                            if (c_j <= 1.02102f) {
                                if (h_w <= 1.13132f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (f_j <= 0.85068f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.85427f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    }
                }
            } else {
                if (c_j <= 1.03409f) {
                    if (h_w <= 1.29554f) {
                        if (f_j <= 0.82178f) {
                            if (f_j <= 0.81753f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.02834f) {
                                    votes[0]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97270f) {
                                if (f_j <= 0.86736f) {
                                    if (h_w <= 1.23593f) {
                                        if (f_j <= 0.83906f) {
                                            if (c_j <= 1.02940f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.96991f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.22068f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.03091f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.24342f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.21830f) {
                                        if (c_j <= 1.03220f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.84763f) {
                                    if (h_w <= 1.18850f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.01816f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.23950f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.02306f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.97337f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.20662f) {
                                            if (h_w <= 1.15789f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (j_w <= 0.97463f) {
                                                if (c_j <= 1.02679f) {
                                                    votes[2]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (f_j <= 0.89823f) {
                        if (h_w <= 1.28937f) {
                            if (f_j <= 0.88827f) {
                                if (c_j <= 1.04378f) {
                                    if (j_w <= 0.96638f) {
                                        if (c_j <= 1.03497f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.96589f) {
                                                if (h_w <= 1.20206f) {
                                                    if (c_j <= 1.04203f) {
                                                        if (c_j <= 1.03956f) {
                                                            if (c_j <= 1.03579f) {
                                                                votes[0]++
                                                            } else {
                                                                if (j_w <= 0.96287f) {
                                                                    votes[4]++
                                                                } else {
                                                                    if (j_w <= 0.96458f) {
                                                                        votes[1]++
                                                                    } else {
                                                                        votes[4]++
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            if (c_j <= 1.04162f) {
                                                                if (h_w <= 1.18664f) {
                                                                    votes[1]++
                                                                } else {
                                                                    if (c_j <= 1.03999f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        votes[3]++
                                                                    }
                                                                }
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (h_w <= 1.25536f) {
                                                        if (c_j <= 1.03879f) {
                                                            if (h_w <= 1.22908f) {
                                                                if (j_w <= 0.96512f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.96062f) {
                                                                if (j_w <= 0.96038f) {
                                                                    if (h_w <= 1.20295f) {
                                                                        votes[1]++
                                                                    } else {
                                                                        if (f_j <= 0.86696f) {
                                                                            votes[0]++
                                                                        } else {
                                                                            votes[3]++
                                                                        }
                                                                    }
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                if (f_j <= 0.86535f) {
                                                                    votes[4]++
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (h_w <= 1.20319f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.21786f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.24933f) {
                                                if (f_j <= 0.88390f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.05814f) {
                                    if (f_j <= 0.89325f) {
                                        if (f_j <= 0.89044f) {
                                            votes[1]++
                                        } else {
                                            if (c_j <= 1.03689f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (j_w <= 0.95426f) {
                            if (c_j <= 1.05720f) {
                                if (c_j <= 1.05240f) {
                                    votes[0]++
                                } else {
                                    if (c_j <= 1.05488f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.94388f) {
                                    if (h_w <= 1.19975f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (j_w <= 0.95587f) {
                                votes[1]++
                            } else {
                                if (j_w <= 0.95736f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.16907f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.24969f) {
                                            if (c_j <= 1.03890f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.96002f) {
                                                    votes[4]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Tree 2
        if (h_w <= 1.15753f) {
            if (j_w <= 0.97883f) {
                if (c_j <= 1.02521f) {
                    if (f_j <= 0.85135f) {
                        votes[3]++
                    } else {
                        if (h_w <= 1.13016f) {
                            votes[1]++
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (j_w <= 0.95761f) {
                        if (f_j <= 0.88426f) {
                            if (j_w <= 0.95210f) {
                                votes[4]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (c_j <= 1.04589f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.89852f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.95530f) {
                                        votes[2]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.12277f) {
                            votes[2]++
                        } else {
                            if (h_w <= 1.12609f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.04316f) {
                                    if (j_w <= 0.96469f) {
                                        if (f_j <= 0.86087f) {
                                            if (j_w <= 0.96277f) {
                                                votes[4]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            if (h_w <= 1.14392f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.03843f) {
                                                    votes[0]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.14471f) {
                                            if (f_j <= 0.88438f) {
                                                if (j_w <= 0.97238f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                }
            } else {
                votes[2]++
            }
        } else {
            if (f_j <= 0.86235f) {
                if (h_w <= 1.19563f) {
                    if (c_j <= 1.03654f) {
                        if (f_j <= 0.85718f) {
                            if (f_j <= 0.81656f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.18198f) {
                                    if (h_w <= 1.17818f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (f_j <= 0.85872f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (j_w <= 0.96176f) {
                            if (c_j <= 1.04203f) {
                                votes[0]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            votes[1]++
                        }
                    }
                } else {
                    if (f_j <= 0.84934f) {
                        if (c_j <= 1.01149f) {
                            if (c_j <= 1.00664f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.80587f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.99290f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.83852f) {
                                if (c_j <= 1.03337f) {
                                    if (f_j <= 0.83584f) {
                                        if (f_j <= 0.80684f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.03316f) {
                                                if (j_w <= 0.97080f) {
                                                    votes[4]++
                                                } else {
                                                    if (j_w <= 0.98571f) {
                                                        if (h_w <= 1.22421f) {
                                                            votes[1]++
                                                        } else {
                                                            if (j_w <= 0.97719f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (c_j <= 1.03632f) {
                                    if (c_j <= 1.01641f) {
                                        if (j_w <= 0.98488f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (f_j <= 0.84267f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.22825f) {
                            if (j_w <= 0.96745f) {
                                if (j_w <= 0.96365f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (c_j <= 1.02639f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.85200f) {
                                        if (j_w <= 0.97024f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.23883f) {
                                votes[2]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (j_w <= 0.95456f) {
                        if (h_w <= 1.27077f) {
                            if (h_w <= 1.17451f) {
                                if (f_j <= 0.90885f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.23957f) {
                                    if (j_w <= 0.94588f) {
                                        votes[2]++
                                    } else {
                                        if (j_w <= 0.95130f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.95211f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.26717f) {
                                        if (j_w <= 0.95227f) {
                                            if (h_w <= 1.25704f) {
                                                if (c_j <= 1.05893f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.88632f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.05717f) {
                                                    votes[1]++
                                                } else {
                                                    if (f_j <= 0.91065f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.07409f) {
                                votes[4]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (h_w <= 1.18431f) {
                            if (j_w <= 0.95853f) {
                                if (h_w <= 1.16611f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.04585f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.86813f) {
                                    if (h_w <= 1.18194f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (f_j <= 0.87989f) {
                                        if (f_j <= 0.87160f) {
                                            votes[1]++
                                        } else {
                                            if (c_j <= 1.02979f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.96840f) {
                                if (h_w <= 1.20091f) {
                                    if (j_w <= 0.96110f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.03732f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03713f) {
                                        if (h_w <= 1.23642f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.23622f) {
                                            if (h_w <= 1.20408f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.95771f) {
                                                    votes[1]++
                                                } else {
                                                    if (j_w <= 0.96038f) {
                                                        votes[0]++
                                                    } else {
                                                        if (f_j <= 0.87217f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.03000f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.23894f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 3
        if (j_w <= 0.97898f) {
            if (h_w <= 1.12243f) {
                if (j_w <= 0.97690f) {
                    votes[2]++
                } else {
                    votes[1]++
                }
            } else {
                if (f_j <= 0.87586f) {
                    if (h_w <= 1.19492f) {
                        if (j_w <= 0.96163f) {
                            if (f_j <= 0.87154f) {
                                if (h_w <= 1.16671f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (c_j <= 1.03473f) {
                                if (h_w <= 1.19118f) {
                                    if (f_j <= 0.87240f) {
                                        if (f_j <= 0.81850f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.02857f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.86813f) {
                                                    if (f_j <= 0.84006f) {
                                                        votes[0]++
                                                    } else {
                                                        if (c_j <= 1.03056f) {
                                                            if (h_w <= 1.16807f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (h_w <= 1.19325f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.14734f) {
                                    if (j_w <= 0.96277f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (j_w <= 0.96405f) {
                                        if (f_j <= 0.85843f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28804f) {
                            if (f_j <= 0.84803f) {
                                if (h_w <= 1.23382f) {
                                    if (j_w <= 0.97160f) {
                                        if (h_w <= 1.20288f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.96956f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (h_w <= 1.25285f) {
                                        if (j_w <= 0.97203f) {
                                            if (j_w <= 0.96773f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03687f) {
                                    if (h_w <= 1.21107f) {
                                        if (c_j <= 1.03274f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03438f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.87421f) {
                                            if (f_j <= 0.86552f) {
                                                if (c_j <= 1.03533f) {
                                                    if (h_w <= 1.22825f) {
                                                        if (h_w <= 1.21399f) {
                                                            votes[0]++
                                                        } else {
                                                            if (c_j <= 1.03067f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        if (h_w <= 1.23883f) {
                                                            votes[2]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04269f) {
                                        if (f_j <= 0.85809f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.96407f) {
                                                if (j_w <= 0.96069f) {
                                                    if (j_w <= 0.96000f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.86432f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                } else {
                    if (h_w <= 1.29114f) {
                        if (j_w <= 0.96266f) {
                            if (h_w <= 1.23639f) {
                                if (f_j <= 0.90489f) {
                                    if (c_j <= 1.04261f) {
                                        if (j_w <= 0.96045f) {
                                            if (f_j <= 0.89110f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (c_j <= 1.04529f) {
                                            if (h_w <= 1.15767f) {
                                                votes[2]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (h_w <= 1.17451f) {
                                                if (h_w <= 1.12592f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (f_j <= 0.89132f) {
                                                    if (c_j <= 1.04905f) {
                                                        if (h_w <= 1.21420f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.05130f) {
                                                        votes[1]++
                                                    } else {
                                                        if (c_j <= 1.05425f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.15521f) {
                                        if (c_j <= 1.04589f) {
                                            votes[2]++
                                        } else {
                                            if (c_j <= 1.04680f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.05748f) {
                                            if (f_j <= 0.92502f) {
                                                if (j_w <= 0.95035f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.20614f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.95227f) {
                                    if (c_j <= 1.06846f) {
                                        if (h_w <= 1.26047f) {
                                            if (h_w <= 1.25629f) {
                                                if (c_j <= 1.06289f) {
                                                    if (c_j <= 1.05893f) {
                                                        if (c_j <= 1.05334f) {
                                                            votes[0]++
                                                        } else {
                                                            if (h_w <= 1.24676f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.95115f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97350f) {
                                if (h_w <= 1.14703f) {
                                    if (f_j <= 0.89377f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        } else {
            if (f_j <= 0.84352f) {
                if (h_w <= 1.16222f) {
                    votes[2]++
                } else {
                    if (f_j <= 0.83768f) {
                        if (h_w <= 1.21012f) {
                            if (j_w <= 0.98638f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.00933f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (f_j <= 0.83087f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[4]++
                    }
                }
            } else {
                if (f_j <= 0.84574f) {
                    votes[1]++
                } else {
                    if (j_w <= 0.98323f) {
                        if (j_w <= 0.97928f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.01929f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[4]++
                    }
                }
            }
        }
        // Tree 4
        if (c_j <= 1.01922f) {
            if (f_j <= 0.80844f) {
                votes[4]++
            } else {
                if (f_j <= 0.83087f) {
                    if (j_w <= 0.98125f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.16180f) {
                            votes[2]++
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (h_w <= 1.15051f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.01686f) {
                            if (h_w <= 1.28220f) {
                                if (f_j <= 0.84112f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (c_j <= 1.01750f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.84533f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.12217f) {
                votes[2]++
            } else {
                if (f_j <= 0.86445f) {
                    if (c_j <= 1.03568f) {
                        if (h_w <= 1.18888f) {
                            if (f_j <= 0.83562f) {
                                if (h_w <= 1.17444f) {
                                    if (c_j <= 1.02043f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (j_w <= 0.97180f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97248f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.97259f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.03109f) {
                                if (j_w <= 0.97503f) {
                                    if (j_w <= 0.97419f) {
                                        if (h_w <= 1.22452f) {
                                            if (j_w <= 0.97368f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (f_j <= 0.84234f) {
                                                votes[4]++
                                            } else {
                                                if (j_w <= 0.97121f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.02603f) {
                                            votes[0]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.85051f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.85457f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97767f) {
                                                votes[3]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.24862f) {
                                    if (f_j <= 0.82832f) {
                                        if (c_j <= 1.03280f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.19365f) {
                            if (j_w <= 0.96173f) {
                                if (j_w <= 0.96064f) {
                                    if (f_j <= 0.85917f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (j_w <= 0.95975f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.21730f) {
                                    if (j_w <= 0.96425f) {
                                        if (j_w <= 0.96287f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (c_j <= 1.03719f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.04213f) {
                        if (j_w <= 0.97180f) {
                            if (c_j <= 1.03254f) {
                                if (h_w <= 1.21429f) {
                                    if (f_j <= 0.86748f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.02984f) {
                                        if (j_w <= 0.97162f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03574f) {
                                    if (j_w <= 0.96732f) {
                                        if (f_j <= 0.89130f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.89505f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.90452f) {
                                                    votes[1]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03291f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.87754f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.15189f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.86581f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.87217f) {
                                                if (c_j <= 1.04165f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.87864f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.96252f) {
                                                        if (c_j <= 1.03958f) {
                                                            if (c_j <= 1.03943f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        if (c_j <= 1.03720f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.15665f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.19939f) {
                                    if (j_w <= 0.97429f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.17374f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.12592f) {
                            votes[0]++
                        } else {
                            if (h_w <= 1.19054f) {
                                if (f_j <= 0.91712f) {
                                    if (f_j <= 0.88955f) {
                                        if (h_w <= 1.16130f) {
                                            if (h_w <= 1.12912f) {
                                                votes[4]++
                                            } else {
                                                if (j_w <= 0.95346f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (j_w <= 0.95256f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.04481f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04727f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.94293f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.16274f) {
                                                votes[2]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.05111f) {
                                    if (h_w <= 1.22832f) {
                                        if (j_w <= 0.95856f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (c_j <= 1.04827f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.95227f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.25900f) {
                                        if (h_w <= 1.23957f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.88632f) {
                                                votes[0]++
                                            } else {
                                                if (c_j <= 1.05893f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.95115f) {
                                            if (h_w <= 1.26243f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.94668f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Tree 5
        if (c_j <= 1.03182f) {
            if (h_w <= 1.14060f) {
                if (j_w <= 0.97497f) {
                    votes[3]++
                } else {
                    if (c_j <= 1.02119f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.02357f) {
                            votes[1]++
                        } else {
                            votes[2]++
                        }
                    }
                }
            } else {
                if (c_j <= 1.02001f) {
                    if (f_j <= 0.85608f) {
                        if (f_j <= 0.83500f) {
                            if (h_w <= 1.20999f) {
                                votes[4]++
                            } else {
                                if (j_w <= 0.98592f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.80584f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.83957f) {
                                votes[4]++
                            } else {
                                if (f_j <= 0.84175f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.98385f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (h_w <= 1.19563f) {
                        if (f_j <= 0.86813f) {
                            if (f_j <= 0.82093f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.19118f) {
                                    if (j_w <= 0.97259f) {
                                        if (c_j <= 1.02830f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (c_j <= 1.02532f) {
                                        if (h_w <= 1.19252f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.02900f) {
                                if (f_j <= 0.87240f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.02639f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.27756f) {
                            if (c_j <= 1.02129f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.21322f) {
                                    if (h_w <= 1.20775f) {
                                        if (f_j <= 0.83432f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.97261f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (f_j <= 0.84968f) {
                                        if (h_w <= 1.23950f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.97345f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.02306f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.23883f) {
                                            if (h_w <= 1.22243f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.85344f) {
                                                    votes[0]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15515f) {
                if (h_w <= 1.12243f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.12450f) {
                        votes[0]++
                    } else {
                        if (j_w <= 0.95729f) {
                            if (h_w <= 1.13309f) {
                                if (c_j <= 1.04908f) {
                                    votes[2]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (f_j <= 0.90308f) {
                                if (j_w <= 0.96174f) {
                                    if (j_w <= 0.95934f) {
                                        if (j_w <= 0.95863f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.96277f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.96565f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.96852f) {
                    if (j_w <= 0.96358f) {
                        if (h_w <= 1.29108f) {
                            if (h_w <= 1.27077f) {
                                if (c_j <= 1.03926f) {
                                    if (h_w <= 1.18621f) {
                                        if (j_w <= 0.96307f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (c_j <= 1.03958f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.88948f) {
                                            if (j_w <= 0.96176f) {
                                                if (f_j <= 0.86343f) {
                                                    if (c_j <= 1.05109f) {
                                                        if (h_w <= 1.16727f) {
                                                            votes[4]++
                                                        } else {
                                                            if (f_j <= 0.85499f) {
                                                                votes[0]++
                                                            } else {
                                                                if (j_w <= 0.95636f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (c_j <= 1.05127f) {
                                                if (c_j <= 1.04370f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.04560f) {
                                                        votes[4]++
                                                    } else {
                                                        if (f_j <= 0.89284f) {
                                                            votes[2]++
                                                        } else {
                                                            if (h_w <= 1.23733f) {
                                                                if (f_j <= 0.91808f) {
                                                                    votes[1]++
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            } else {
                                                                if (f_j <= 0.90717f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[4]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.05299f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.05541f) {
                                                        votes[1]++
                                                    } else {
                                                        if (h_w <= 1.26185f) {
                                                            if (j_w <= 0.94685f) {
                                                                if (h_w <= 1.21248f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (f_j <= 0.92724f) {
                                                                        if (j_w <= 0.94429f) {
                                                                            votes[1]++
                                                                        } else {
                                                                            votes[3]++
                                                                        }
                                                                    } else {
                                                                        votes[0]++
                                                                    }
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.07409f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (c_j <= 1.03508f) {
                            if (h_w <= 1.22301f) {
                                votes[0]++
                            } else {
                                if (j_w <= 0.96750f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.88073f) {
                                        if (h_w <= 1.23493f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.23750f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.96810f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.19365f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.96887f) {
                        if (f_j <= 0.88578f) {
                            votes[3]++
                        } else {
                            votes[0]++
                        }
                    } else {
                        votes[0]++
                    }
                }
            }
        }
        // Tree 6
        if (h_w <= 1.14943f) {
            if (h_w <= 1.12457f) {
                votes[2]++
            } else {
                if (c_j <= 1.04960f) {
                    if (f_j <= 0.89354f) {
                        if (j_w <= 0.96277f) {
                            if (j_w <= 0.95735f) {
                                votes[2]++
                            } else {
                                if (c_j <= 1.04040f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (h_w <= 1.12609f) {
                                votes[3]++
                            } else {
                                if (j_w <= 0.97539f) {
                                    votes[2]++
                                } else {
                                    if (f_j <= 0.83567f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (c_j <= 1.05113f) {
                        votes[3]++
                    } else {
                        votes[4]++
                    }
                }
            }
        } else {
            if (h_w <= 1.22195f) {
                if (c_j <= 1.03574f) {
                    if (c_j <= 1.03091f) {
                        if (f_j <= 0.87944f) {
                            if (c_j <= 1.02998f) {
                                if (h_w <= 1.16521f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.84995f) {
                                        if (h_w <= 1.19722f) {
                                            if (j_w <= 0.97158f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97452f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.97738f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.84600f) {
                                                if (j_w <= 0.98942f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.80587f) {
                                                        votes[3]++
                                                    } else {
                                                        if (c_j <= 1.00715f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.17545f) {
                                            if (h_w <= 1.17399f) {
                                                if (f_j <= 0.86350f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.18135f) {
                                    if (c_j <= 1.03041f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (j_w <= 0.96643f) {
                            if (c_j <= 1.03518f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (f_j <= 0.81324f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.19638f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.03274f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.20206f) {
                        if (h_w <= 1.15628f) {
                            if (j_w <= 0.95861f) {
                                if (h_w <= 1.15308f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (f_j <= 0.92502f) {
                                if (h_w <= 1.15857f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.17039f) {
                                        if (f_j <= 0.90834f) {
                                            if (h_w <= 1.15923f) {
                                                votes[2]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (c_j <= 1.04463f) {
                                            if (h_w <= 1.18425f) {
                                                if (j_w <= 0.95854f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.87643f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.03672f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.86100f) {
                                                        if (f_j <= 0.85140f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        if (j_w <= 0.95962f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.17739f) {
                                                if (j_w <= 0.95048f) {
                                                    if (f_j <= 0.91208f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (h_w <= 1.18598f) {
                                                    votes[3]++
                                                } else {
                                                    if (j_w <= 0.95234f) {
                                                        votes[0]++
                                                    } else {
                                                        if (f_j <= 0.86996f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.18366f) {
                                    votes[1]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.85923f) {
                            if (c_j <= 1.03786f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (f_j <= 0.91482f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.92174f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.30735f) {
                    if (f_j <= 0.85053f) {
                        if (c_j <= 1.01279f) {
                            votes[0]++
                        } else {
                            if (c_j <= 1.03528f) {
                                if (h_w <= 1.24870f) {
                                    if (j_w <= 0.97857f) {
                                        if (c_j <= 1.03335f) {
                                            if (h_w <= 1.24253f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (h_w <= 1.22670f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (j_w <= 0.96436f) {
                            if (h_w <= 1.27019f) {
                                if (j_w <= 0.95227f) {
                                    if (c_j <= 1.05299f) {
                                        if (h_w <= 1.23123f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.87974f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.24082f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.88632f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.94193f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.25911f) {
                                                        votes[0]++
                                                    } else {
                                                        if (j_w <= 0.94594f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.89012f) {
                                        if (c_j <= 1.04125f) {
                                            if (f_j <= 0.86498f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (c_j <= 1.02900f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.86167f) {
                                    if (c_j <= 1.03139f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 7
        if (h_w <= 1.14897f) {
            if (j_w <= 0.95275f) {
                if (j_w <= 0.94982f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.95135f) {
                        votes[4]++
                    } else {
                        votes[3]++
                    }
                }
            } else {
                if (h_w <= 1.12243f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.03747f) {
                        if (j_w <= 0.96568f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.12920f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.97198f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.12433f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.96026f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.84841f) {
                                    votes[2]++
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (c_j <= 1.03574f) {
                if (h_w <= 1.19740f) {
                    if (f_j <= 0.87184f) {
                        if (f_j <= 0.83884f) {
                            if (c_j <= 1.02490f) {
                                votes[3]++
                            } else {
                                if (f_j <= 0.80838f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97616f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.17477f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.16159f) {
                            votes[1]++
                        } else {
                            if (f_j <= 0.87494f) {
                                votes[4]++
                            } else {
                                if (j_w <= 0.97207f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84981f) {
                        if (c_j <= 1.03316f) {
                            if (h_w <= 1.20053f) {
                                if (f_j <= 0.83450f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (j_w <= 0.99058f) {
                                    if (j_w <= 0.97270f) {
                                        if (j_w <= 0.97157f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96767f) {
                                if (h_w <= 1.24594f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (j_w <= 0.97075f) {
                            if (j_w <= 0.96621f) {
                                if (j_w <= 0.96596f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (h_w <= 1.20718f) {
                                    if (h_w <= 1.20031f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (f_j <= 0.85716f) {
                                        if (f_j <= 0.85380f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (j_w <= 0.96760f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.86356f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.96852f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.86345f) {
                                if (f_j <= 0.85357f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.21440f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.88277f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.88337f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f_j <= 0.86445f) {
                    if (j_w <= 0.95910f) {
                        if (j_w <= 0.95140f) {
                            votes[1]++
                        } else {
                            if (f_j <= 0.86026f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (f_j <= 0.85239f) {
                            if (h_w <= 1.23143f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.85786f) {
                                if (h_w <= 1.18970f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.89068f) {
                        if (j_w <= 0.93699f) {
                            votes[4]++
                        } else {
                            if (c_j <= 1.05354f) {
                                if (c_j <= 1.04067f) {
                                    if (h_w <= 1.19361f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.03695f) {
                                            if (j_w <= 0.96463f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04323f) {
                                        if (c_j <= 1.04212f) {
                                            if (h_w <= 1.21001f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.86581f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.96062f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.88437f) {
                                            if (c_j <= 1.05021f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.95138f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.23022f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.24397f) {
                                    if (f_j <= 0.88661f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.95304f) {
                            if (j_w <= 0.94968f) {
                                if (f_j <= 0.90482f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.06219f) {
                                        if (h_w <= 1.27019f) {
                                            if (f_j <= 0.92182f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.94540f) {
                                                    if (c_j <= 1.05886f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[2]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (c_j <= 1.06695f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.07576f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.89413f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (c_j <= 1.04696f) {
                                if (c_j <= 1.04460f) {
                                    if (c_j <= 1.03908f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.95746f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.90988f) {
                                                if (j_w <= 0.96193f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.91001f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            }
        }
        // Tree 8
        if (h_w <= 1.15515f) {
            if (f_j <= 0.80758f) {
                votes[3]++
            } else {
                if (c_j <= 1.04589f) {
                    if (f_j <= 0.84445f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.97656f) {
                            if (f_j <= 0.87587f) {
                                if (h_w <= 1.15124f) {
                                    if (c_j <= 1.03824f) {
                                        votes[2]++
                                    } else {
                                        if (c_j <= 1.04116f) {
                                            votes[4]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (j_w <= 0.95841f) {
                                    if (f_j <= 0.90074f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (c_j <= 1.01807f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.95562f) {
                        if (f_j <= 0.89149f) {
                            if (h_w <= 1.13931f) {
                                if (f_j <= 0.88528f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.88821f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        votes[0]++
                    }
                }
            }
        } else {
            if (h_w <= 1.30735f) {
                if (f_j <= 0.86445f) {
                    if (h_w <= 1.19198f) {
                        if (f_j <= 0.82093f) {
                            votes[0]++
                        } else {
                            if (c_j <= 1.03877f) {
                                if (h_w <= 1.19118f) {
                                    if (c_j <= 1.02979f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03093f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.16727f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.01149f) {
                            if (h_w <= 1.21012f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (c_j <= 1.02538f) {
                                if (f_j <= 0.83852f) {
                                    if (j_w <= 0.98571f) {
                                        if (h_w <= 1.22670f) {
                                            votes[1]++
                                        } else {
                                            if (c_j <= 1.02267f) {
                                                votes[3]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (j_w <= 0.97432f) {
                                    if (f_j <= 0.84673f) {
                                        if (f_j <= 0.82988f) {
                                            if (c_j <= 1.03397f) {
                                                if (f_j <= 0.82208f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.82775f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (c_j <= 1.02660f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.84097f) {
                                                    if (j_w <= 0.96774f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03087f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.23634f) {
                                                if (f_j <= 0.85505f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.86419f) {
                                                        if (j_w <= 0.96121f) {
                                                            if (j_w <= 0.95975f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.25877f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.97463f) {
                                        votes[2]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.88802f) {
                        if (c_j <= 1.03981f) {
                            if (h_w <= 1.22959f) {
                                if (j_w <= 0.96576f) {
                                    if (j_w <= 0.96298f) {
                                        votes[3]++
                                    } else {
                                        if (h_w <= 1.19361f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.96968f) {
                                        if (f_j <= 0.87341f) {
                                            if (j_w <= 0.96832f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (j_w <= 0.97429f) {
                                            if (j_w <= 0.97205f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (h_w <= 1.18059f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.28821f) {
                                    if (j_w <= 0.96857f) {
                                        if (h_w <= 1.25995f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (j_w <= 0.95835f) {
                                if (h_w <= 1.24933f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (c_j <= 1.04173f) {
                                    if (f_j <= 0.86930f) {
                                        if (j_w <= 0.96038f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.04370f) {
                            if (h_w <= 1.22571f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.89123f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (c_j <= 1.04460f) {
                                votes[4]++
                            } else {
                                if (j_w <= 0.95349f) {
                                    if (f_j <= 0.91402f) {
                                        if (h_w <= 1.18546f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.24372f) {
                                                if (f_j <= 0.89621f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.22795f) {
                                                        if (h_w <= 1.20627f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.24688f) {
                                                    votes[4]++
                                                } else {
                                                    if (j_w <= 0.94603f) {
                                                        if (c_j <= 1.06165f) {
                                                            votes[4]++
                                                        } else {
                                                            if (f_j <= 0.88875f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.94806f) {
                                            if (f_j <= 0.92476f) {
                                                votes[0]++
                                            } else {
                                                if (c_j <= 1.06926f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.89657f) {
                                        votes[2]++
                                    } else {
                                        if (c_j <= 1.04617f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                votes[1]++
            }
        }
        // Tree 9
        if (j_w <= 0.97891f) {
            if (c_j <= 1.03676f) {
                if (h_w <= 1.20141f) {
                    if (f_j <= 0.86833f) {
                        if (f_j <= 0.82093f) {
                            if (f_j <= 0.80838f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (f_j <= 0.85182f) {
                                if (h_w <= 1.07846f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (j_w <= 0.96545f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.15871f) {
                                        if (j_w <= 0.97496f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.02464f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.16515f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97343f) {
                                                if (f_j <= 0.86432f) {
                                                    votes[3]++
                                                } else {
                                                    if (c_j <= 1.03409f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.96599f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.14680f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.87184f) {
                                    if (f_j <= 0.86973f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.97429f) {
                                        if (h_w <= 1.16629f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.02685f) {
                        if (h_w <= 1.23950f) {
                            if (c_j <= 1.02566f) {
                                votes[4]++
                            } else {
                                if (c_j <= 1.02633f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (c_j <= 1.02178f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (f_j <= 0.84667f) {
                            if (h_w <= 1.23750f) {
                                if (j_w <= 0.96770f) {
                                    votes[1]++
                                } else {
                                    if (c_j <= 1.03022f) {
                                        if (h_w <= 1.21743f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (f_j <= 0.88658f) {
                                if (f_j <= 0.85200f) {
                                    votes[0]++
                                } else {
                                    if (c_j <= 1.03652f) {
                                        if (h_w <= 1.27681f) {
                                            if (j_w <= 0.96702f) {
                                                if (j_w <= 0.96611f) {
                                                    if (j_w <= 0.96543f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (j_w <= 0.96913f) {
                                                    votes[3]++
                                                } else {
                                                    if (j_w <= 0.97186f) {
                                                        if (f_j <= 0.86934f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.30229f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.95962f) {
                    if (c_j <= 1.06675f) {
                        if (f_j <= 0.90322f) {
                            if (f_j <= 0.89768f) {
                                if (j_w <= 0.95883f) {
                                    if (h_w <= 1.17770f) {
                                        if (f_j <= 0.89196f) {
                                            if (c_j <= 1.04410f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        if (j_w <= 0.94900f) {
                                            if (j_w <= 0.94397f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (f_j <= 0.86360f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.04925f) {
                                                    if (j_w <= 0.95552f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.17442f) {
                                        votes[2]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.94957f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.95264f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.14952f) {
                                if (c_j <= 1.04680f) {
                                    if (c_j <= 1.04589f) {
                                        votes[2]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (j_w <= 0.95456f) {
                                    if (h_w <= 1.15534f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.91015f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.93183f) {
                                                if (f_j <= 0.92476f) {
                                                    if (c_j <= 1.05479f) {
                                                        if (j_w <= 0.95189f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.05720f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95606f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.92123f) {
                                            if (f_j <= 0.91430f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.07576f) {
                            if (j_w <= 0.93699f) {
                                if (h_w <= 1.27285f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (h_w <= 1.15382f) {
                        if (c_j <= 1.03915f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.86391f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (h_w <= 1.31054f) {
                            if (f_j <= 0.84267f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.04072f) {
                                    if (h_w <= 1.19841f) {
                                        if (j_w <= 0.96227f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.88045f) {
                                                if (f_j <= 0.86588f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96407f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.25372f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.22241f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.23130f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.16240f) {
                if (f_j <= 0.87120f) {
                    votes[2]++
                } else {
                    votes[3]++
                }
            } else {
                if (c_j <= 1.01149f) {
                    if (c_j <= 1.00747f) {
                        votes[0]++
                    } else {
                        if (j_w <= 0.99065f) {
                            votes[0]++
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (j_w <= 0.98076f) {
                        if (h_w <= 1.26868f) {
                            votes[3]++
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (f_j <= 0.83644f) {
                            if (f_j <= 0.81193f) {
                                votes[4]++
                            } else {
                                if (h_w <= 1.20297f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.22421f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.98385f) {
                                votes[4]++
                            } else {
                                if (f_j <= 0.84112f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun evaluateTrees10to19(
        h_w: Float,
        j_w: Float,
        f_j: Float,
        c_j: Float,
        votes: IntArray
    ) {
        // Tree 10
        if (h_w <= 1.15408f) {
            if (c_j <= 1.02119f) {
                if (c_j <= 1.01807f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.98200f) {
                        votes[2]++
                    } else {
                        votes[3]++
                    }
                }
            } else {
                if (j_w <= 0.96867f) {
                    if (h_w <= 1.12254f) {
                        votes[2]++
                    } else {
                        if (f_j <= 0.90308f) {
                            if (c_j <= 1.05031f) {
                                if (h_w <= 1.13858f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.96038f) {
                                        votes[2]++
                                    } else {
                                        if (h_w <= 1.14717f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.88611f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    }
                } else {
                    if (c_j <= 1.02232f) {
                        votes[1]++
                    } else {
                        if (c_j <= 1.02559f) {
                            if (h_w <= 1.12944f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[3]++
                        }
                    }
                }
            }
        } else {
            if (f_j <= 0.84963f) {
                if (h_w <= 1.18756f) {
                    if (c_j <= 1.02745f) {
                        votes[3]++
                    } else {
                        if (f_j <= 0.84343f) {
                            votes[0]++
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (f_j <= 0.83852f) {
                        if (c_j <= 1.00851f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.96770f) {
                                if (c_j <= 1.03733f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (j_w <= 0.96897f) {
                                    if (f_j <= 0.83129f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (h_w <= 1.20032f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.83091f) {
                                            if (j_w <= 0.98942f) {
                                                if (c_j <= 1.01535f) {
                                                    votes[4]++
                                                } else {
                                                    if (c_j <= 1.02336f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (h_w <= 1.21221f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97784f) {
                                                    votes[1]++
                                                } else {
                                                    if (f_j <= 0.83497f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.97206f) {
                            if (j_w <= 0.96882f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[4]++
                        }
                    }
                }
            } else {
                if (h_w <= 1.28900f) {
                    if (c_j <= 1.03676f) {
                        if (f_j <= 0.87840f) {
                            if (c_j <= 1.03581f) {
                                if (h_w <= 1.24360f) {
                                    if (c_j <= 1.03565f) {
                                        if (j_w <= 0.98066f) {
                                            if (f_j <= 0.85884f) {
                                                if (f_j <= 0.85716f) {
                                                    if (c_j <= 1.02680f) {
                                                        if (c_j <= 1.02411f) {
                                                            votes[3]++
                                                        } else {
                                                            if (h_w <= 1.21097f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[2]++
                                                            }
                                                        }
                                                    } else {
                                                        if (h_w <= 1.22068f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (f_j <= 0.86809f) {
                                                    votes[3]++
                                                } else {
                                                    if (h_w <= 1.17706f) {
                                                        if (c_j <= 1.02876f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        if (f_j <= 0.86982f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (h_w <= 1.19559f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (h_w <= 1.23114f) {
                                if (c_j <= 1.02866f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (c_j <= 1.04474f) {
                            if (j_w <= 0.95965f) {
                                if (h_w <= 1.20065f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.88967f) {
                                    if (f_j <= 0.86581f) {
                                        if (c_j <= 1.04097f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (f_j <= 0.87217f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.96060f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.03835f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.87314f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.20089f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.95566f) {
                                if (j_w <= 0.95426f) {
                                    if (h_w <= 1.23659f) {
                                        if (h_w <= 1.17688f) {
                                            if (f_j <= 0.90885f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (f_j <= 0.92643f) {
                                                if (f_j <= 0.91808f) {
                                                    if (j_w <= 0.94639f) {
                                                        votes[3]++
                                                    } else {
                                                        if (j_w <= 0.95130f) {
                                                            if (f_j <= 0.87026f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.95416f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.26185f) {
                                            if (j_w <= 0.95227f) {
                                                if (h_w <= 1.25094f) {
                                                    if (c_j <= 1.06289f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (h_w <= 1.25704f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04696f) {
                                        votes[2]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 11
        if (j_w <= 0.96450f) {
            if (h_w <= 1.15515f) {
                if (h_w <= 1.12694f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.03921f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.95934f) {
                            if (c_j <= 1.04316f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.92493f) {
                                    if (f_j <= 0.88528f) {
                                        if (h_w <= 1.14013f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (c_j <= 1.04960f) {
                                            if (c_j <= 1.04662f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            votes[3]++
                        }
                    }
                }
            } else {
                if (c_j <= 1.05541f) {
                    if (j_w <= 0.96358f) {
                        if (f_j <= 0.87718f) {
                            if (c_j <= 1.03892f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.04072f) {
                                    if (j_w <= 0.96195f) {
                                        if (j_w <= 0.96099f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.83309f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.84267f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.05230f) {
                                        if (c_j <= 1.04378f) {
                                            if (j_w <= 0.95962f) {
                                                if (h_w <= 1.20006f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (f_j <= 0.86581f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.04100f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.25704f) {
                                if (c_j <= 1.03890f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.95426f) {
                                        if (h_w <= 1.17688f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.24270f) {
                                                if (j_w <= 0.94801f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.05127f) {
                                                        if (f_j <= 0.91808f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.95130f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.04041f) {
                                            if (h_w <= 1.20635f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (h_w <= 1.16127f) {
                                                votes[2]++
                                            } else {
                                                if (f_j <= 0.89798f) {
                                                    if (h_w <= 1.17670f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (c_j <= 1.03687f) {
                            votes[0]++
                        } else {
                            votes[1]++
                        }
                    }
                } else {
                    if (h_w <= 1.26243f) {
                        if (j_w <= 0.94685f) {
                            if (c_j <= 1.06664f) {
                                if (c_j <= 1.06219f) {
                                    if (j_w <= 0.94378f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.94564f) {
                                            votes[2]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        votes[4]++
                    }
                }
            }
        } else {
            if (j_w <= 0.97898f) {
                if (j_w <= 0.96463f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.22390f) {
                        if (f_j <= 0.89264f) {
                            if (f_j <= 0.81656f) {
                                if (h_w <= 1.16884f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (j_w <= 0.97041f) {
                                    if (h_w <= 1.21056f) {
                                        if (j_w <= 0.96546f) {
                                            if (h_w <= 1.15849f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.21176f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.87184f) {
                                        if (c_j <= 1.02232f) {
                                            if (h_w <= 1.13731f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (h_w <= 1.12920f) {
                                                votes[2]++
                                            } else {
                                                if (h_w <= 1.18114f) {
                                                    if (j_w <= 0.97264f) {
                                                        if (j_w <= 0.97248f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.97368f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.21322f) {
                                                            if (f_j <= 0.84779f) {
                                                                if (j_w <= 0.97452f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (h_w <= 1.19490f) {
                                                                        votes[3]++
                                                                    } else {
                                                                        votes[4]++
                                                                    }
                                                                }
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.02612f) {
                                            votes[1]++
                                        } else {
                                            if (c_j <= 1.02993f) {
                                                if (c_j <= 1.02686f) {
                                                    votes[3]++
                                                } else {
                                                    if (h_w <= 1.14772f) {
                                                        votes[2]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (h_w <= 1.28907f) {
                            if (j_w <= 0.97746f) {
                                if (f_j <= 0.84667f) {
                                    if (c_j <= 1.02492f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.80906f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.23033f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.97182f) {
                                        if (c_j <= 1.03283f) {
                                            if (h_w <= 1.24457f) {
                                                if (h_w <= 1.22935f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (j_w <= 0.96918f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.25736f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (c_j <= 1.02758f) {
                                            votes[2]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            } else {
                if (f_j <= 0.83768f) {
                    if (h_w <= 1.15030f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.01950f) {
                            if (f_j <= 0.83087f) {
                                if (f_j <= 0.81193f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (j_w <= 0.98803f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (j_w <= 0.98385f) {
                        if (j_w <= 0.98019f) {
                            votes[2]++
                        } else {
                            if (h_w <= 1.17848f) {
                                votes[2]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (h_w <= 1.18129f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.01536f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            }
        }
        // Tree 12
        if (h_w <= 1.15124f) {
            if (j_w <= 0.97784f) {
                if (h_w <= 1.12243f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.03747f) {
                        votes[3]++
                    } else {
                        if (j_w <= 0.95210f) {
                            if (h_w <= 1.12592f) {
                                votes[0]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (f_j <= 0.91498f) {
                                if (f_j <= 0.89730f) {
                                    if (c_j <= 1.04116f) {
                                        if (c_j <= 1.03867f) {
                                            votes[2]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                votes[2]++
            }
        } else {
            if (j_w <= 0.97089f) {
                if (j_w <= 0.95795f) {
                    if (h_w <= 1.29108f) {
                        if (h_w <= 1.27077f) {
                            if (f_j <= 0.88920f) {
                                if (j_w <= 0.95219f) {
                                    if (f_j <= 0.88047f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.91514f) {
                                    if (h_w <= 1.19145f) {
                                        if (j_w <= 0.95183f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.04560f) {
                                                votes[4]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.89229f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.95363f) {
                                                if (c_j <= 1.05001f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.25985f) {
                                                        votes[0]++
                                                    } else {
                                                        if (c_j <= 1.05588f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95456f) {
                                        if (h_w <= 1.25768f) {
                                            if (f_j <= 0.93183f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (j_w <= 0.95606f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.93106f) {
                                votes[0]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (h_w <= 1.29485f) {
                        if (h_w <= 1.23688f) {
                            if (f_j <= 0.81933f) {
                                if (f_j <= 0.80343f) {
                                    votes[1]++
                                } else {
                                    if (j_w <= 0.96922f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.90483f) {
                                    if (c_j <= 1.03037f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.17716f) {
                                            if (j_w <= 0.96010f) {
                                                if (j_w <= 0.95889f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.03283f) {
                                                if (f_j <= 0.86467f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.87380f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.21003f) {
                                                    if (j_w <= 0.96549f) {
                                                        if (j_w <= 0.96545f) {
                                                            if (f_j <= 0.86212f) {
                                                                if (h_w <= 1.19841f) {
                                                                    votes[1]++
                                                                } else {
                                                                    if (f_j <= 0.85241f) {
                                                                        votes[4]++
                                                                    } else {
                                                                        votes[1]++
                                                                    }
                                                                }
                                                            } else {
                                                                if (f_j <= 0.88195f) {
                                                                    votes[4]++
                                                                } else {
                                                                    if (c_j <= 1.03992f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        votes[1]++
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (h_w <= 1.22083f) {
                                                        if (j_w <= 0.96278f) {
                                                            votes[0]++
                                                        } else {
                                                            if (h_w <= 1.21176f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    } else {
                                                        if (j_w <= 0.96038f) {
                                                            votes[0]++
                                                        } else {
                                                            if (j_w <= 0.96715f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96415f) {
                                votes[4]++
                            } else {
                                if (h_w <= 1.26847f) {
                                    if (h_w <= 1.25919f) {
                                        if (h_w <= 1.24309f) {
                                            if (f_j <= 0.85195f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (h_w <= 1.19648f) {
                    if (f_j <= 0.82093f) {
                        votes[0]++
                    } else {
                        if (h_w <= 1.19118f) {
                            if (f_j <= 0.86289f) {
                                if (j_w <= 0.97259f) {
                                    if (j_w <= 0.97191f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.97222f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.85729f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (j_w <= 0.97429f) {
                                    if (h_w <= 1.16509f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (h_w <= 1.17374f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.84289f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84981f) {
                        if (j_w <= 0.98864f) {
                            if (f_j <= 0.83891f) {
                                if (f_j <= 0.83091f) {
                                    if (j_w <= 0.97719f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.01535f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.83810f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.98385f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    } else {
                        if (j_w <= 0.97134f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.85898f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.20297f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.88337f) {
                                        if (h_w <= 1.29946f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Tree 13
        if (h_w <= 1.15432f) {
            if (h_w <= 1.12254f) {
                if (j_w <= 0.97690f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.02055f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (f_j <= 0.89453f) {
                    if (f_j <= 0.83985f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.04238f) {
                            if (f_j <= 0.86063f) {
                                if (c_j <= 1.03226f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (c_j <= 1.04882f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.88528f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                } else {
                    votes[2]++
                }
            }
        } else {
            if (j_w <= 0.96224f) {
                if (h_w <= 1.29108f) {
                    if (f_j <= 0.87724f) {
                        if (f_j <= 0.85941f) {
                            if (h_w <= 1.19563f) {
                                if (j_w <= 0.95967f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.96176f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (j_w <= 0.95858f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.86533f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.04125f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.21891f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.92759f) {
                            if (j_w <= 0.96193f) {
                                if (h_w <= 1.27019f) {
                                    if (j_w <= 0.95813f) {
                                        if (h_w <= 1.19019f) {
                                            if (j_w <= 0.94661f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.88803f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.05884f) {
                                                if (c_j <= 1.05127f) {
                                                    if (f_j <= 0.90831f) {
                                                        if (j_w <= 0.95552f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        if (h_w <= 1.23217f) {
                                                            if (j_w <= 0.95418f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    if (j_w <= 0.94968f) {
                                                        if (h_w <= 1.22414f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.93863f) {
                                                    votes[1]++
                                                } else {
                                                    if (f_j <= 0.90319f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (c_j <= 1.05194f) {
                                if (h_w <= 1.18855f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.93183f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.24078f) {
                                        votes[2]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (f_j <= 0.84782f) {
                    if (h_w <= 1.19563f) {
                        if (c_j <= 1.02652f) {
                            if (f_j <= 0.81684f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (c_j <= 1.02891f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (f_j <= 0.82988f) {
                            if (j_w <= 0.96770f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.20255f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.24995f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.98260f) {
                                if (c_j <= 1.02986f) {
                                    if (h_w <= 1.23950f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.97746f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.22005f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.22939f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.23322f) {
                        if (f_j <= 0.85884f) {
                            if (h_w <= 1.15885f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.22825f) {
                                    if (h_w <= 1.21399f) {
                                        if (j_w <= 0.97528f) {
                                            if (j_w <= 0.96900f) {
                                                if (f_j <= 0.85654f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.22306f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (f_j <= 0.86460f) {
                                votes[3]++
                            } else {
                                if (f_j <= 0.87279f) {
                                    if (c_j <= 1.03137f) {
                                        if (h_w <= 1.17374f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.20317f) {
                                            if (h_w <= 1.18982f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.16062f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.18027f) {
                                            if (h_w <= 1.17457f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (c_j <= 1.03694f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.88918f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.97182f) {
                            if (c_j <= 1.03478f) {
                                if (f_j <= 0.86167f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.25778f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.96830f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.29092f) {
                                    if (f_j <= 0.86878f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 14
        if (c_j <= 1.03667f) {
            if (h_w <= 1.12092f) {
                if (c_j <= 1.02072f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.97690f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (f_j <= 0.83077f) {
                    if (j_w <= 0.97193f) {
                        if (h_w <= 1.17769f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.82208f) {
                                if (c_j <= 1.03280f) {
                                    if (c_j <= 1.03108f) {
                                        votes[4]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.17314f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.82715f) {
                                if (f_j <= 0.81115f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.24360f) {
                        if (h_w <= 1.17401f) {
                            if (f_j <= 0.89717f) {
                                if (j_w <= 0.97118f) {
                                    if (f_j <= 0.86657f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96970f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (h_w <= 1.17488f) {
                                votes[4]++
                            } else {
                                if (c_j <= 1.02685f) {
                                    if (h_w <= 1.19484f) {
                                        if (h_w <= 1.18858f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.19245f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.01816f) {
                                            if (j_w <= 0.98803f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.02566f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.84717f) {
                                                    votes[4]++
                                                } else {
                                                    if (j_w <= 0.97458f) {
                                                        votes[2]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03291f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96728f) {
                                            if (h_w <= 1.18728f) {
                                                votes[0]++
                                            } else {
                                                if (h_w <= 1.20783f) {
                                                    votes[3]++
                                                } else {
                                                    if (h_w <= 1.21004f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.30735f) {
                            if (c_j <= 1.02530f) {
                                if (h_w <= 1.25873f) {
                                    if (f_j <= 0.84112f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.25125f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.86027f) {
                                        votes[4]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15521f) {
                if (f_j <= 0.89151f) {
                    if (f_j <= 0.88636f) {
                        if (c_j <= 1.05031f) {
                            if (h_w <= 1.13117f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.14167f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.96038f) {
                                        votes[2]++
                                    } else {
                                        if (j_w <= 0.96174f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.84841f) {
                                                votes[2]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        votes[3]++
                    }
                } else {
                    if (h_w <= 1.11829f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.05402f) {
                            if (h_w <= 1.13048f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[3]++
                        }
                    }
                }
            } else {
                if (h_w <= 1.29108f) {
                    if (h_w <= 1.27077f) {
                        if (h_w <= 1.20306f) {
                            if (c_j <= 1.04204f) {
                                if (h_w <= 1.18486f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.18621f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03811f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.87470f) {
                                                if (h_w <= 1.19563f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.91528f) {
                                    if (j_w <= 0.95730f) {
                                        if (f_j <= 0.89950f) {
                                            if (c_j <= 1.04738f) {
                                                votes[2]++
                                            } else {
                                                if (c_j <= 1.05005f) {
                                                    if (c_j <= 1.04822f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.04982f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (c_j <= 1.05299f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.94290f) {
                                            votes[0]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.03687f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.03854f) {
                                    votes[1]++
                                } else {
                                    if (j_w <= 0.95552f) {
                                        if (h_w <= 1.23323f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.23855f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.95181f) {
                                                    if (c_j <= 1.06289f) {
                                                        if (h_w <= 1.26047f) {
                                                            if (f_j <= 0.91979f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.94594f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    } else {
                                                        if (h_w <= 1.25071f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.05001f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96064f) {
                                            if (h_w <= 1.21902f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (c_j <= 1.03929f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.93106f) {
                            votes[0]++
                        } else {
                            votes[4]++
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 15
        if (j_w <= 0.97897f) {
            if (h_w <= 1.14917f) {
                if (j_w <= 0.97698f) {
                    if (f_j <= 0.88339f) {
                        if (f_j <= 0.85534f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.03805f) {
                                if (c_j <= 1.03151f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.86983f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.88247f) {
                                        votes[2]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.12243f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.89354f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.12952f) {
                                    votes[0]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.97874f) {
                        votes[1]++
                    } else {
                        votes[3]++
                    }
                }
            } else {
                if (j_w <= 0.96449f) {
                    if (h_w <= 1.29927f) {
                        if (c_j <= 1.03976f) {
                            if (c_j <= 1.03958f) {
                                if (c_j <= 1.03727f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.89780f) {
                                        if (h_w <= 1.22732f) {
                                            if (c_j <= 1.03926f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.96266f) {
                                            votes[2]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (c_j <= 1.05795f) {
                                if (h_w <= 1.18755f) {
                                    if (c_j <= 1.04162f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.18252f) {
                                            if (h_w <= 1.17376f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.88434f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.89726f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95130f) {
                                        if (c_j <= 1.05352f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.88852f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.87499f) {
                                            if (f_j <= 0.86670f) {
                                                if (j_w <= 0.96128f) {
                                                    if (h_w <= 1.23130f) {
                                                        if (f_j <= 0.85909f) {
                                                            votes[4]++
                                                        } else {
                                                            if (h_w <= 1.19745f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (h_w <= 1.21001f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.95181f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.95349f) {
                                                    if (f_j <= 0.90717f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.95426f) {
                                                        if (j_w <= 0.95418f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.06195f) {
                                    if (j_w <= 0.94293f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (h_w <= 1.27285f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (h_w <= 1.19563f) {
                        if (f_j <= 0.86395f) {
                            if (h_w <= 1.18581f) {
                                if (j_w <= 0.97259f) {
                                    if (j_w <= 0.97248f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (h_w <= 1.19252f) {
                                    if (j_w <= 0.97343f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (h_w <= 1.16671f) {
                                if (h_w <= 1.15665f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.96545f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.18728f) {
                                        if (c_j <= 1.03158f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28907f) {
                            if (f_j <= 0.84849f) {
                                if (f_j <= 0.82988f) {
                                    if (f_j <= 0.82832f) {
                                        if (c_j <= 1.03008f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.96769f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.96986f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.19919f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.03560f) {
                                        if (c_j <= 1.02900f) {
                                            if (j_w <= 0.97439f) {
                                                if (j_w <= 0.97195f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (c_j <= 1.02609f) {
                                                    if (c_j <= 1.02587f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.20654f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.96611f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.25778f) {
                                                        votes[0]++
                                                    } else {
                                                        if (j_w <= 0.96822f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.16222f) {
                if (h_w <= 1.12092f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.13132f) {
                        votes[3]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (f_j <= 0.83143f) {
                    if (h_w <= 1.22241f) {
                        if (f_j <= 0.79735f) {
                            votes[3]++
                        } else {
                            votes[4]++
                        }
                    } else {
                        votes[0]++
                    }
                } else {
                    if (h_w <= 1.19642f) {
                        votes[3]++
                    } else {
                        if (h_w <= 1.29877f) {
                            votes[4]++
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 16
        if (h_w <= 1.15135f) {
            if (h_w <= 1.12261f) {
                votes[2]++
            } else {
                if (j_w <= 0.95135f) {
                    if (h_w <= 1.12592f) {
                        votes[0]++
                    } else {
                        votes[4]++
                    }
                } else {
                    if (c_j <= 1.04188f) {
                        if (f_j <= 0.83985f) {
                            votes[2]++
                        } else {
                            if (j_w <= 0.96529f) {
                                if (h_w <= 1.14183f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (f_j <= 0.87956f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.13612f) {
                            if (h_w <= 1.12770f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[2]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.22046f) {
                if (c_j <= 1.03707f) {
                    if (j_w <= 0.97310f) {
                        if (j_w <= 0.97076f) {
                            if (j_w <= 0.97053f) {
                                if (h_w <= 1.21056f) {
                                    if (j_w <= 0.96546f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.03059f) {
                                            if (h_w <= 1.17579f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (f_j <= 0.86842f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.03408f) {
                                                    if (j_w <= 0.96881f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03567f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (c_j <= 1.02236f) {
                            if (h_w <= 1.19642f) {
                                if (j_w <= 0.98066f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.01633f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.98125f) {
                                            votes[1]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.01069f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (f_j <= 0.82981f) {
                                if (h_w <= 1.16672f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (c_j <= 1.02645f) {
                                    if (h_w <= 1.18858f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.84779f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.95727f) {
                        if (h_w <= 1.19510f) {
                            if (f_j <= 0.88826f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.06121f) {
                                    if (c_j <= 1.05686f) {
                                        if (c_j <= 1.05128f) {
                                            if (h_w <= 1.16611f) {
                                                votes[2]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (f_j <= 0.88914f) {
                                if (j_w <= 0.95101f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (c_j <= 1.04204f) {
                            if (f_j <= 0.85399f) {
                                if (c_j <= 1.03976f) {
                                    if (c_j <= 1.03956f) {
                                        if (f_j <= 0.84267f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.87078f) {
                                    if (c_j <= 1.03875f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (h_w <= 1.15521f) {
                                        votes[2]++
                                    } else {
                                        if (f_j <= 0.90600f) {
                                            if (h_w <= 1.21042f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (j_w <= 0.96096f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.15572f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.20550f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (f_j <= 0.86383f) {
                        if (f_j <= 0.83129f) {
                            if (f_j <= 0.80852f) {
                                votes[1]++
                            } else {
                                if (c_j <= 1.03397f) {
                                    if (h_w <= 1.24995f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (f_j <= 0.84803f) {
                                votes[4]++
                            } else {
                                if (j_w <= 0.96415f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.23942f) {
                                        if (c_j <= 1.02832f) {
                                            votes[2]++
                                        } else {
                                            if (c_j <= 1.03091f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03478f) {
                            if (j_w <= 0.96849f) {
                                if (j_w <= 0.96730f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.22845f) {
                                votes[1]++
                            } else {
                                if (j_w <= 0.95237f) {
                                    if (j_w <= 0.94968f) {
                                        if (f_j <= 0.88584f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.26243f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.28582f) {
                                                    if (f_j <= 0.90920f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.25704f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.05063f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03746f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.89012f) {
                                            if (j_w <= 0.95552f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.01915f) {
                        votes[4]++
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 17
        if (j_w <= 0.98401f) {
            if (c_j <= 1.03531f) {
                if (f_j <= 0.84981f) {
                    if (h_w <= 1.19563f) {
                        if (f_j <= 0.82093f) {
                            if (f_j <= 0.80594f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.14086f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (f_j <= 0.83891f) {
                            if (c_j <= 1.03337f) {
                                if (f_j <= 0.83395f) {
                                    if (c_j <= 1.02336f) {
                                        if (h_w <= 1.22421f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.21520f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.82988f) {
                                                if (f_j <= 0.82372f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.23765f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (c_j <= 1.01641f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.87812f) {
                        if (c_j <= 1.02232f) {
                            if (f_j <= 0.87451f) {
                                if (j_w <= 0.97928f) {
                                    if (c_j <= 1.02165f) {
                                        votes[2]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (j_w <= 0.97233f) {
                                if (h_w <= 1.19674f) {
                                    if (c_j <= 1.03059f) {
                                        if (c_j <= 1.02977f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03041f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (f_j <= 0.85878f) {
                                        if (j_w <= 0.96957f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97028f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03169f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.97264f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.97337f) {
                                        if (f_j <= 0.86775f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (c_j <= 1.02443f) {
                                            if (c_j <= 1.02347f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (h_w <= 1.22442f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.02902f) {
                            votes[1]++
                        } else {
                            if (c_j <= 1.03443f) {
                                if (h_w <= 1.18607f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                if (c_j <= 1.04420f) {
                    if (h_w <= 1.24229f) {
                        if (h_w <= 1.15767f) {
                            if (h_w <= 1.12296f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.86360f) {
                                    if (f_j <= 0.86087f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (c_j <= 1.03943f) {
                                        if (c_j <= 1.03775f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.19087f) {
                                if (j_w <= 0.95965f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.18049f) {
                                        if (c_j <= 1.03940f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.96266f) {
                                    if (c_j <= 1.04072f) {
                                        if (h_w <= 1.20078f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (h_w <= 1.21487f) {
                                            if (j_w <= 0.95902f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.87204f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.19365f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28697f) {
                            if (j_w <= 0.96415f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[1]++
                        }
                    }
                } else {
                    if (f_j <= 0.90290f) {
                        if (j_w <= 0.94799f) {
                            if (h_w <= 1.14151f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.28937f) {
                                    if (f_j <= 0.89354f) {
                                        if (f_j <= 0.88875f) {
                                            if (c_j <= 1.06554f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (j_w <= 0.95130f) {
                                if (f_j <= 0.88611f) {
                                    if (f_j <= 0.88256f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (c_j <= 1.05370f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.14856f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.05071f) {
                                    if (j_w <= 0.95754f) {
                                        if (h_w <= 1.16639f) {
                                            if (c_j <= 1.04882f) {
                                                votes[2]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (h_w <= 1.23022f) {
                                                if (j_w <= 0.95727f) {
                                                    if (f_j <= 0.89597f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (j_w <= 0.95350f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.19439f) {
                            if (h_w <= 1.11639f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.12433f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.94233f) {
                                        votes[0]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.91047f) {
                                if (j_w <= 0.94429f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (c_j <= 1.05479f) {
                                    if (j_w <= 0.95418f) {
                                        if (c_j <= 1.05130f) {
                                            if (j_w <= 0.95349f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.94590f) {
                                        if (c_j <= 1.06926f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15032f) {
                votes[2]++
            } else {
                if (h_w <= 1.20999f) {
                    if (f_j <= 0.81902f) {
                        votes[4]++
                    } else {
                        votes[3]++
                    }
                } else {
                    votes[0]++
                }
            }
        }
        // Tree 18
        if (h_w <= 1.12694f) {
            if (h_w <= 1.12243f) {
                if (j_w <= 0.97690f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.02055f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (h_w <= 1.12378f) {
                    votes[0]++
                } else {
                    if (f_j <= 0.90895f) {
                        if (c_j <= 1.02743f) {
                            votes[2]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            }
        } else {
            if (j_w <= 0.96433f) {
                if (h_w <= 1.29108f) {
                    if (j_w <= 0.95420f) {
                        if (h_w <= 1.15657f) {
                            if (j_w <= 0.95135f) {
                                if (c_j <= 1.05593f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.15305f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (h_w <= 1.23855f) {
                                if (h_w <= 1.17688f) {
                                    if (c_j <= 1.05577f) {
                                        votes[4]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (c_j <= 1.05665f) {
                                        if (f_j <= 0.88184f) {
                                            if (h_w <= 1.19764f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.24082f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.92873f) {
                                        if (h_w <= 1.26185f) {
                                            if (c_j <= 1.05120f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.88632f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.94567f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.94751f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.15923f) {
                            if (c_j <= 1.04169f) {
                                if (f_j <= 0.87361f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.89711f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.90641f) {
                                            votes[1]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.13723f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (c_j <= 1.04587f) {
                                if (h_w <= 1.17514f) {
                                    if (j_w <= 0.95965f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.04202f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95730f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.04173f) {
                                            if (h_w <= 1.20078f) {
                                                if (c_j <= 1.03999f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (j_w <= 0.96064f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.90285f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.04370f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.22221f) {
                    if (j_w <= 0.98086f) {
                        if (h_w <= 1.21266f) {
                            if (f_j <= 0.82093f) {
                                if (j_w <= 0.97384f) {
                                    if (c_j <= 1.03167f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.97924f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.89843f) {
                                    if (c_j <= 1.02858f) {
                                        if (f_j <= 0.88077f) {
                                            if (h_w <= 1.14060f) {
                                                if (f_j <= 0.83694f) {
                                                    votes[2]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.84584f) {
                                                    votes[3]++
                                                } else {
                                                    if (h_w <= 1.19484f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.19842f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (f_j <= 0.84054f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.85744f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.96509f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.85913f) {
                                                        votes[0]++
                                                    } else {
                                                        if (c_j <= 1.03206f) {
                                                            if (f_j <= 0.86784f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            if (h_w <= 1.19197f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (c_j <= 1.02703f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (h_w <= 1.14488f) {
                            votes[2]++
                        } else {
                            if (h_w <= 1.20999f) {
                                if (h_w <= 1.17811f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (f_j <= 0.80587f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.30735f) {
                        if (f_j <= 0.87488f) {
                            if (c_j <= 1.01279f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.23034f) {
                                    if (f_j <= 0.83746f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.02458f) {
                                            votes[4]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.85175f) {
                                        if (j_w <= 0.96790f) {
                                            if (f_j <= 0.81472f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.24862f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.01641f) {
                                                if (h_w <= 1.25458f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.86250f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.28907f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 19
        if (h_w <= 1.15485f) {
            if (h_w <= 1.11926f) {
                if (h_w <= 1.11008f) {
                    votes[2]++
                } else {
                    if (f_j <= 0.87781f) {
                        if (j_w <= 0.98137f) {
                            votes[1]++
                        } else {
                            votes[2]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (f_j <= 0.89210f) {
                    if (j_w <= 0.96388f) {
                        if (f_j <= 0.88611f) {
                            if (f_j <= 0.84841f) {
                                votes[2]++
                            } else {
                                if (c_j <= 1.05031f) {
                                    if (c_j <= 1.04384f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (c_j <= 1.04778f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (c_j <= 1.02102f) {
                            if (h_w <= 1.12430f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (h_w <= 1.12433f) {
                        votes[0]++
                    } else {
                        votes[2]++
                    }
                }
            }
        } else {
            if (c_j <= 1.03676f) {
                if (h_w <= 1.19170f) {
                    if (j_w <= 0.96549f) {
                        votes[0]++
                    } else {
                        if (f_j <= 0.86784f) {
                            if (j_w <= 0.97431f) {
                                if (c_j <= 1.02890f) {
                                    if (h_w <= 1.18272f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (f_j <= 0.85774f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03082f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (j_w <= 0.97310f) {
                                if (c_j <= 1.02900f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.96814f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97429f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.97270f) {
                        if (f_j <= 0.87809f) {
                            if (h_w <= 1.30183f) {
                                if (h_w <= 1.25919f) {
                                    if (f_j <= 0.83129f) {
                                        if (j_w <= 0.96770f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.19341f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.83680f) {
                                                votes[4]++
                                            } else {
                                                if (h_w <= 1.22390f) {
                                                    if (j_w <= 0.96728f) {
                                                        if (h_w <= 1.20783f) {
                                                            votes[3]++
                                                        } else {
                                                            if (c_j <= 1.03581f) {
                                                                if (f_j <= 0.86290f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    } else {
                                                        if (c_j <= 1.03268f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    if (h_w <= 1.23617f) {
                                                        votes[0]++
                                                    } else {
                                                        if (c_j <= 1.03209f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.89581f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (f_j <= 0.84934f) {
                            if (j_w <= 0.98864f) {
                                if (c_j <= 1.02129f) {
                                    if (j_w <= 0.98044f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.01771f) {
                                            if (j_w <= 0.98488f) {
                                                if (f_j <= 0.84646f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (j_w <= 0.98135f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.22983f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (h_w <= 1.19513f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.20207f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.97478f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.02115f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f_j <= 0.84267f) {
                    votes[0]++
                } else {
                    if (j_w <= 0.96436f) {
                        if (h_w <= 1.30088f) {
                            if (j_w <= 0.95727f) {
                                if (c_j <= 1.05374f) {
                                    if (f_j <= 0.89994f) {
                                        if (h_w <= 1.16639f) {
                                            votes[2]++
                                        } else {
                                            if (j_w <= 0.95130f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.87724f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.95552f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.26717f) {
                                            if (j_w <= 0.95363f) {
                                                if (h_w <= 1.25670f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.92919f) {
                                        if (j_w <= 0.93542f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.94847f) {
                                                if (c_j <= 1.06248f) {
                                                    if (h_w <= 1.25196f) {
                                                        if (c_j <= 1.05884f) {
                                                            if (h_w <= 1.19167f) {
                                                                votes[4]++
                                                            } else {
                                                                if (j_w <= 0.94718f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.06554f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.06937f) {
                                            if (j_w <= 0.94580f) {
                                                votes[2]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.04370f) {
                                    if (h_w <= 1.23168f) {
                                        if (f_j <= 0.85909f) {
                                            if (h_w <= 1.19279f) {
                                                if (f_j <= 0.85096f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (f_j <= 0.87017f) {
                                                if (f_j <= 0.86201f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.03859f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.96060f) {
                                                    if (f_j <= 0.88195f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.96136f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.19542f) {
                                                            if (h_w <= 1.17239f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        votes[0]++
                    }
                }
            }
        }
    }

    private fun evaluateTrees20to29(
        h_w: Float,
        j_w: Float,
        f_j: Float,
        c_j: Float,
        votes: IntArray
    ) {
        // Tree 20
        if (j_w <= 0.98043f) {
            if (h_w <= 1.15135f) {
                if (f_j <= 0.84656f) {
                    votes[3]++
                } else {
                    if (f_j <= 0.88339f) {
                        if (j_w <= 0.95210f) {
                            votes[4]++
                        } else {
                            if (c_j <= 1.02365f) {
                                if (h_w <= 1.10739f) {
                                    votes[2]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.96467f) {
                                    votes[2]++
                                } else {
                                    if (h_w <= 1.12209f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.89453f) {
                            if (h_w <= 1.12261f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.95380f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (c_j <= 1.04644f) {
                                if (h_w <= 1.12433f) {
                                    votes[0]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.96036f) {
                    if (h_w <= 1.23855f) {
                        if (h_w <= 1.17039f) {
                            if (f_j <= 0.90834f) {
                                if (h_w <= 1.15923f) {
                                    votes[2]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.15534f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.92179f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.04420f) {
                                if (j_w <= 0.95912f) {
                                    if (h_w <= 1.19399f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.86432f) {
                                    votes[1]++
                                } else {
                                    if (c_j <= 1.05393f) {
                                        if (c_j <= 1.04886f) {
                                            if (j_w <= 0.95418f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.89505f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.21916f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.21248f) {
                                            if (j_w <= 0.94825f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.94663f) {
                            if (j_w <= 0.93542f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.30662f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (c_j <= 1.05012f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.29554f) {
                        if (f_j <= 0.84295f) {
                            if (f_j <= 0.82824f) {
                                if (h_w <= 1.21913f) {
                                    if (f_j <= 0.82093f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.19530f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.80906f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97595f) {
                                    if (h_w <= 1.20712f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (f_j <= 0.84125f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.21056f) {
                                if (c_j <= 1.03565f) {
                                    if (h_w <= 1.15924f) {
                                        if (j_w <= 0.96899f) {
                                            votes[3]++
                                        } else {
                                            if (j_w <= 0.97671f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.02449f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97565f) {
                                                if (h_w <= 1.16097f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.89269f) {
                                                        if (c_j <= 1.02738f) {
                                                            votes[3]++
                                                        } else {
                                                            if (j_w <= 0.97270f) {
                                                                if (h_w <= 1.16394f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[3]++
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.02487f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.19365f) {
                                        if (j_w <= 0.96209f) {
                                            if (c_j <= 1.03975f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.16486f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.03871f) {
                                                if (j_w <= 0.96413f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (f_j <= 0.90727f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.21786f) {
                                    if (j_w <= 0.96556f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.96407f) {
                                        if (f_j <= 0.86498f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.96118f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96849f) {
                                            if (c_j <= 1.03437f) {
                                                votes[3]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (c_j <= 1.02900f) {
                                                if (c_j <= 1.02591f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (h_w <= 1.24457f) {
                                                    if (h_w <= 1.22935f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        } else {
            if (h_w <= 1.16240f) {
                if (h_w <= 1.14488f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.01633f) {
                        votes[3]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (h_w <= 1.19166f) {
                    if (f_j <= 0.85646f) {
                        votes[3]++
                    } else {
                        votes[1]++
                    }
                } else {
                    if (c_j <= 1.01149f) {
                        votes[0]++
                    } else {
                        if (h_w <= 1.28220f) {
                            if (j_w <= 0.98488f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            votes[4]++
                        }
                    }
                }
            }
        }
        // Tree 21
        if (c_j <= 1.02064f) {
            if (c_j <= 1.00790f) {
                if (c_j <= 1.00715f) {
                    votes[3]++
                } else {
                    votes[0]++
                }
            } else {
                if (f_j <= 0.84352f) {
                    if (f_j <= 0.80844f) {
                        if (f_j <= 0.80458f) {
                            if (c_j <= 1.01189f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (h_w <= 1.14753f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.82786f) {
                                votes[3]++
                            } else {
                                if (f_j <= 0.83500f) {
                                    if (c_j <= 1.01429f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.98135f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.20454f) {
                                            votes[2]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84574f) {
                        votes[1]++
                    } else {
                        if (j_w <= 0.98158f) {
                            votes[1]++
                        } else {
                            if (c_j <= 1.01776f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.01807f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15385f) {
                if (j_w <= 0.97817f) {
                    if (f_j <= 0.86336f) {
                        if (h_w <= 1.12816f) {
                            votes[2]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (j_w <= 0.95193f) {
                            if (h_w <= 1.11846f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.94901f) {
                                    if (c_j <= 1.05666f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (c_j <= 1.02615f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.12217f) {
                                    votes[2]++
                                } else {
                                    if (h_w <= 1.13739f) {
                                        if (c_j <= 1.04369f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (c_j <= 1.02999f) {
                        if (h_w <= 1.19490f) {
                            if (f_j <= 0.82178f) {
                                votes[0]++
                            } else {
                                if (j_w <= 0.97526f) {
                                    if (f_j <= 0.87240f) {
                                        if (f_j <= 0.84006f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (j_w <= 0.97429f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (c_j <= 1.02703f) {
                                if (c_j <= 1.02190f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.02561f) {
                                        if (f_j <= 0.83877f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.86664f) {
                                            if (h_w <= 1.21322f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.97214f) {
                                    if (c_j <= 1.02900f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.97134f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.27453f) {
                            if (c_j <= 1.03035f) {
                                votes[1]++
                            } else {
                                if (j_w <= 0.96383f) {
                                    if (f_j <= 0.89044f) {
                                        if (h_w <= 1.17674f) {
                                            if (c_j <= 1.04203f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.86229f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.88390f) {
                                                if (h_w <= 1.21370f) {
                                                    if (c_j <= 1.05439f) {
                                                        if (f_j <= 0.86492f) {
                                                            votes[1]++
                                                        } else {
                                                            if (c_j <= 1.04008f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.04072f) {
                                                        if (j_w <= 0.96170f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        if (j_w <= 0.95138f) {
                                                            votes[0]++
                                                        } else {
                                                            if (h_w <= 1.22656f) {
                                                                votes[0]++
                                                            } else {
                                                                if (f_j <= 0.86581f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.04655f) {
                                                    if (f_j <= 0.88871f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96239f) {
                                            if (j_w <= 0.95730f) {
                                                if (f_j <= 0.89284f) {
                                                    votes[2]++
                                                } else {
                                                    if (c_j <= 1.04877f) {
                                                        if (j_w <= 0.95606f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        if (h_w <= 1.23957f) {
                                                            if (f_j <= 0.92786f) {
                                                                if (c_j <= 1.05127f) {
                                                                    votes[1]++
                                                                } else {
                                                                    if (h_w <= 1.17039f) {
                                                                        if (c_j <= 1.05948f) {
                                                                            votes[4]++
                                                                        } else {
                                                                            votes[0]++
                                                                        }
                                                                    } else {
                                                                        if (f_j <= 0.90108f) {
                                                                            votes[0]++
                                                                        } else {
                                                                            if (f_j <= 0.91386f) {
                                                                                votes[3]++
                                                                            } else {
                                                                                votes[0]++
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                votes[2]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.94968f) {
                                                                if (c_j <= 1.05488f) {
                                                                    votes[1]++
                                                                } else {
                                                                    if (h_w <= 1.26243f) {
                                                                        if (c_j <= 1.05893f) {
                                                                            votes[0]++
                                                                        } else {
                                                                            votes[1]++
                                                                        }
                                                                    } else {
                                                                        votes[4]++
                                                                    }
                                                                }
                                                            } else {
                                                                if (j_w <= 0.95237f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[4]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.23322f) {
                                        if (f_j <= 0.85878f) {
                                            if (h_w <= 1.20234f) {
                                                if (h_w <= 1.18171f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (j_w <= 0.96556f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.03574f) {
                                                if (j_w <= 0.96758f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.86842f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                if (f_j <= 0.86590f) {
                                                    if (j_w <= 0.96540f) {
                                                        votes[3]++
                                                    } else {
                                                        if (c_j <= 1.03579f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.23750f) {
                                            if (j_w <= 0.96770f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (f_j <= 0.86876f) {
                                                if (j_w <= 0.96876f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 22
        if (j_w <= 0.96549f) {
            if (f_j <= 0.87217f) {
                if (f_j <= 0.86581f) {
                    if (f_j <= 0.86445f) {
                        if (h_w <= 1.22135f) {
                            if (h_w <= 1.16727f) {
                                if (j_w <= 0.95938f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.96277f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03626f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.84604f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.17939f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.86273f) {
                                                if (j_w <= 0.96417f) {
                                                    if (h_w <= 1.19380f) {
                                                        if (c_j <= 1.04382f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        votes[0]++
                    }
                } else {
                    if (h_w <= 1.20394f) {
                        if (c_j <= 1.04344f) {
                            votes[3]++
                        } else {
                            votes[0]++
                        }
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (j_w <= 0.94883f) {
                    if (f_j <= 0.90286f) {
                        if (h_w <= 1.14151f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.88370f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.05541f) {
                                    if (c_j <= 1.05478f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.06796f) {
                            if (c_j <= 1.05783f) {
                                if (c_j <= 1.05737f) {
                                    if (f_j <= 0.91187f) {
                                        if (f_j <= 0.90637f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (c_j <= 1.06295f) {
                                    if (c_j <= 1.06053f) {
                                        votes[2]++
                                    } else {
                                        if (c_j <= 1.06224f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.06477f) {
                                        votes[0]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.92960f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.15238f) {
                        if (f_j <= 0.91169f) {
                            if (h_w <= 1.12323f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.95617f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.95913f) {
                                        if (c_j <= 1.04339f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.91498f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        }
                    } else {
                        if (c_j <= 1.05119f) {
                            if (c_j <= 1.05063f) {
                                if (j_w <= 0.95275f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.95998f) {
                                        if (c_j <= 1.04529f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.04875f) {
                                                if (h_w <= 1.21109f) {
                                                    if (h_w <= 1.18179f) {
                                                        votes[2]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96060f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.22733f) {
                                                if (c_j <= 1.03854f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.88879f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.16640f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (f_j <= 0.88796f) {
                                                    if (c_j <= 1.03666f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                }
            }
        } else {
            if (c_j <= 1.02147f) {
                if (f_j <= 0.84352f) {
                    if (f_j <= 0.83903f) {
                        if (h_w <= 1.14753f) {
                            votes[2]++
                        } else {
                            if (h_w <= 1.19166f) {
                                votes[3]++
                            } else {
                                if (j_w <= 0.98864f) {
                                    if (c_j <= 1.01581f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.98045f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.82422f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (f_j <= 0.86604f) {
                        if (j_w <= 0.98381f) {
                            votes[1]++
                        } else {
                            if (j_w <= 0.98385f) {
                                votes[4]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (f_j <= 0.84816f) {
                    if (h_w <= 1.19490f) {
                        votes[3]++
                    } else {
                        if (j_w <= 0.97193f) {
                            if (c_j <= 1.03008f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.23750f) {
                                    if (c_j <= 1.03397f) {
                                        if (f_j <= 0.83760f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            votes[4]++
                        }
                    }
                } else {
                    if (h_w <= 1.24289f) {
                        if (f_j <= 0.88057f) {
                            if (h_w <= 1.12358f) {
                                votes[2]++
                            } else {
                                if (c_j <= 1.03056f) {
                                    if (h_w <= 1.13885f) {
                                        votes[2]++
                                    } else {
                                        if (c_j <= 1.03001f) {
                                            if (f_j <= 0.85918f) {
                                                if (h_w <= 1.22442f) {
                                                    if (f_j <= 0.85736f) {
                                                        if (j_w <= 0.97610f) {
                                                            votes[3]++
                                                        } else {
                                                            if (j_w <= 0.97646f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[2]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.03031f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.22367f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96888f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.97198f) {
                                if (h_w <= 1.16486f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (c_j <= 1.03180f) {
                            if (j_w <= 0.97180f) {
                                if (h_w <= 1.25736f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.88837f) {
                                if (c_j <= 1.03308f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            }
        }
        // Tree 23
        if (j_w <= 0.96463f) {
            if (h_w <= 1.15408f) {
                if (f_j <= 0.89295f) {
                    if (f_j <= 0.88636f) {
                        if (j_w <= 0.95210f) {
                            votes[4]++
                        } else {
                            if (h_w <= 1.15031f) {
                                if (h_w <= 1.11175f) {
                                    votes[2]++
                                } else {
                                    if (f_j <= 0.86995f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (c_j <= 1.05223f) {
                            votes[3]++
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (j_w <= 0.95562f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.95612f) {
                            votes[0]++
                        } else {
                            votes[2]++
                        }
                    }
                }
            } else {
                if (c_j <= 1.05737f) {
                    if (c_j <= 1.04475f) {
                        if (f_j <= 0.85633f) {
                            if (c_j <= 1.04203f) {
                                if (f_j <= 0.84267f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.84747f) {
                                        votes[4]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (h_w <= 1.28620f) {
                                if (c_j <= 1.04370f) {
                                    if (f_j <= 0.90483f) {
                                        if (c_j <= 1.03833f) {
                                            if (f_j <= 0.87948f) {
                                                if (j_w <= 0.96383f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.03727f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (j_w <= 0.96087f) {
                                                if (h_w <= 1.20006f) {
                                                    if (h_w <= 1.18698f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.96062f) {
                                                        if (h_w <= 1.23130f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.20467f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.04420f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.89754f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.25704f) {
                            if (f_j <= 0.89950f) {
                                if (j_w <= 0.95130f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.88067f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.04655f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.04877f) {
                                    if (h_w <= 1.21109f) {
                                        if (f_j <= 0.90351f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.94685f) {
                                        if (f_j <= 0.91015f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.95181f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                } else {
                    votes[4]++
                }
            }
        } else {
            if (h_w <= 1.12209f) {
                if (j_w <= 0.97880f) {
                    if (j_w <= 0.97690f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (h_w <= 1.17401f) {
                    if (f_j <= 0.89951f) {
                        if (j_w <= 0.97977f) {
                            if (f_j <= 0.86813f) {
                                if (h_w <= 1.12944f) {
                                    if (h_w <= 1.12690f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (h_w <= 1.15665f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (h_w <= 1.14753f) {
                                if (f_j <= 0.85467f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (h_w <= 1.30816f) {
                        if (f_j <= 0.83033f) {
                            if (j_w <= 0.97247f) {
                                if (f_j <= 0.82208f) {
                                    if (f_j <= 0.80343f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.81243f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (c_j <= 1.02806f) {
                                if (h_w <= 1.20142f) {
                                    if (f_j <= 0.89156f) {
                                        if (j_w <= 0.98284f) {
                                            if (h_w <= 1.20000f) {
                                                if (c_j <= 1.02258f) {
                                                    if (c_j <= 1.01962f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.97565f) {
                                                        if (h_w <= 1.17706f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (f_j <= 0.85211f) {
                                        if (c_j <= 1.01641f) {
                                            if (h_w <= 1.25458f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (h_w <= 1.23950f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.84261f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.87763f) {
                                    if (h_w <= 1.23888f) {
                                        if (c_j <= 1.03113f) {
                                            if (h_w <= 1.18039f) {
                                                if (f_j <= 0.84680f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (f_j <= 0.86101f) {
                                                if (c_j <= 1.03567f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.86548f) {
                                                    if (f_j <= 0.86387f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.03198f) {
                                                        votes[0]++
                                                    } else {
                                                        if (j_w <= 0.96570f) {
                                                            votes[1]++
                                                        } else {
                                                            if (f_j <= 0.87021f) {
                                                                if (c_j <= 1.03383f) {
                                                                    votes[4]++
                                                                } else {
                                                                    votes[3]++
                                                                }
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96876f) {
                                            if (f_j <= 0.85266f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.28907f) {
                                        if (c_j <= 1.03283f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.88837f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 24
        if (c_j <= 1.02147f) {
            if (j_w <= 0.99258f) {
                if (h_w <= 1.16240f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.19642f) {
                        votes[3]++
                    } else {
                        if (h_w <= 1.32638f) {
                            if (c_j <= 1.01149f) {
                                if (j_w <= 0.98963f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.83500f) {
                                    if (c_j <= 1.01451f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.98385f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.84112f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            } else {
                if (f_j <= 0.83037f) {
                    votes[0]++
                } else {
                    votes[3]++
                }
            }
        } else {
            if (h_w <= 1.15283f) {
                if (f_j <= 0.90205f) {
                    if (h_w <= 1.12791f) {
                        if (c_j <= 1.02357f) {
                            votes[1]++
                        } else {
                            if (h_w <= 1.12234f) {
                                votes[2]++
                            } else {
                                if (c_j <= 1.04003f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03747f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.88528f) {
                                if (c_j <= 1.05031f) {
                                    if (h_w <= 1.15149f) {
                                        if (f_j <= 0.86087f) {
                                            if (c_j <= 1.03867f) {
                                                votes[2]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (c_j <= 1.03924f) {
                    if (h_w <= 1.22221f) {
                        if (c_j <= 1.02765f) {
                            if (f_j <= 0.87740f) {
                                if (j_w <= 0.97747f) {
                                    if (f_j <= 0.82093f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.20092f) {
                                            if (j_w <= 0.97335f) {
                                                votes[4]++
                                            } else {
                                                if (j_w <= 0.97563f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.83383f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.84600f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.81934f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (c_j <= 1.03426f) {
                                if (f_j <= 0.84006f) {
                                    if (j_w <= 0.96922f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.02940f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.03076f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.16394f) {
                                        if (h_w <= 1.15943f) {
                                            if (f_j <= 0.86597f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97182f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (j_w <= 0.96830f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.03871f) {
                                    if (j_w <= 0.96417f) {
                                        if (h_w <= 1.19077f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (f_j <= 0.85856f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03579f) {
                                                if (f_j <= 0.87040f) {
                                                    if (h_w <= 1.16996f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.86570f) {
                            if (h_w <= 1.25347f) {
                                if (f_j <= 0.82832f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.84766f) {
                                        if (f_j <= 0.82988f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.85308f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.85664f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (h_w <= 1.28907f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.20006f) {
                        if (c_j <= 1.04463f) {
                            if (j_w <= 0.95967f) {
                                votes[4]++
                            } else {
                                if (f_j <= 0.87759f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (f_j <= 0.89950f) {
                                if (h_w <= 1.16639f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (c_j <= 1.05640f) {
                                    if (h_w <= 1.17653f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.90936f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.18261f) {
                                        votes[0]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.92942f) {
                            if (h_w <= 1.29334f) {
                                if (c_j <= 1.04072f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.24372f) {
                                        if (c_j <= 1.04100f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.21031f) {
                                                if (j_w <= 0.95187f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.20308f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.23342f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.23855f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.05120f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.05541f) {
                                                if (j_w <= 0.94968f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                }
            }
        }
        // Tree 25
        if (h_w <= 1.12217f) {
            votes[2]++
        } else {
            if (h_w <= 1.30735f) {
                if (h_w <= 1.19597f) {
                    if (j_w <= 0.96558f) {
                        if (j_w <= 0.94661f) {
                            if (j_w <= 0.94233f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.05748f) {
                                    votes[0]++
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (f_j <= 0.86445f) {
                                if (f_j <= 0.85370f) {
                                    if (f_j <= 0.84293f) {
                                        votes[2]++
                                    } else {
                                        if (j_w <= 0.96298f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04234f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.95133f) {
                                    if (h_w <= 1.12592f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.17485f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.15408f) {
                                        if (f_j <= 0.90081f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.12433f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.88766f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.89000f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.04481f) {
                                                    votes[4]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.96943f) {
                            if (f_j <= 0.89092f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (f_j <= 0.87944f) {
                                if (j_w <= 0.98118f) {
                                    if (h_w <= 1.17747f) {
                                        if (c_j <= 1.02688f) {
                                            if (h_w <= 1.12920f) {
                                                votes[2]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (f_j <= 0.87184f) {
                                                if (f_j <= 0.85934f) {
                                                    if (f_j <= 0.85349f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.87311f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.84100f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.19027f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97530f) {
                                                    votes[3]++
                                                } else {
                                                    if (h_w <= 1.19245f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.12430f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84934f) {
                        if (j_w <= 0.98864f) {
                            if (j_w <= 0.96986f) {
                                if (f_j <= 0.84097f) {
                                    if (f_j <= 0.80343f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (c_j <= 1.02379f) {
                                    if (f_j <= 0.82777f) {
                                        if (j_w <= 0.98490f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (j_w <= 0.97590f) {
                                        if (c_j <= 1.02888f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.23016f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    } else {
                        if (j_w <= 0.96362f) {
                            if (h_w <= 1.27593f) {
                                if (f_j <= 0.86343f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.20078f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.86581f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.06664f) {
                                                if (h_w <= 1.22656f) {
                                                    if (h_w <= 1.21588f) {
                                                        if (f_j <= 0.90444f) {
                                                            votes[1]++
                                                        } else {
                                                            if (c_j <= 1.04401f) {
                                                                votes[1]++
                                                            } else {
                                                                if (c_j <= 1.05060f) {
                                                                    votes[3]++
                                                                } else {
                                                                    if (f_j <= 0.91015f) {
                                                                        votes[3]++
                                                                    } else {
                                                                        votes[0]++
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (h_w <= 1.23502f) {
                                                        votes[1]++
                                                    } else {
                                                        if (f_j <= 0.88047f) {
                                                            votes[1]++
                                                        } else {
                                                            if (c_j <= 1.05001f) {
                                                                votes[4]++
                                                            } else {
                                                                if (j_w <= 0.95024f) {
                                                                    if (f_j <= 0.88632f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        if (f_j <= 0.92580f) {
                                                                            if (j_w <= 0.94231f) {
                                                                                if (c_j <= 1.06165f) {
                                                                                    votes[4]++
                                                                                } else {
                                                                                    votes[1]++
                                                                                }
                                                                            } else {
                                                                                votes[1]++
                                                                            }
                                                                        } else {
                                                                            votes[0]++
                                                                        }
                                                                    }
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (j_w <= 0.96849f) {
                                if (h_w <= 1.20268f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.86728f) {
                                        if (j_w <= 0.96611f) {
                                            if (c_j <= 1.03698f) {
                                                if (j_w <= 0.96531f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97075f) {
                                    if (f_j <= 0.87953f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.23894f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.84995f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.87628f) {
                                            if (h_w <= 1.22267f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97316f) {
                                                    votes[1]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.02921f) {
                                                if (f_j <= 0.88337f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                votes[1]++
            }
        }
        // Tree 26
        if (c_j <= 1.03667f) {
            if (h_w <= 1.14086f) {
                if (f_j <= 0.85711f) {
                    if (c_j <= 1.02567f) {
                        votes[2]++
                    } else {
                        votes[3]++
                    }
                } else {
                    if (h_w <= 1.11672f) {
                        if (j_w <= 0.97880f) {
                            votes[1]++
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (f_j <= 0.89951f) {
                            votes[3]++
                        } else {
                            votes[2]++
                        }
                    }
                }
            } else {
                if (h_w <= 1.21698f) {
                    if (f_j <= 0.81656f) {
                        if (j_w <= 0.98327f) {
                            if (h_w <= 1.16884f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (f_j <= 0.86408f) {
                            if (c_j <= 1.02498f) {
                                if (h_w <= 1.19118f) {
                                    if (j_w <= 0.98069f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (c_j <= 1.01505f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.84981f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.85918f) {
                                    if (f_j <= 0.85774f) {
                                        if (h_w <= 1.21146f) {
                                            if (j_w <= 0.97191f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97222f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.84488f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03567f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (h_w <= 1.16988f) {
                                if (f_j <= 0.86813f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.02900f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.87571f) {
                                    if (h_w <= 1.19341f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.19638f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.18810f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.85053f) {
                        if (c_j <= 1.01279f) {
                            votes[0]++
                        } else {
                            if (f_j <= 0.82988f) {
                                if (c_j <= 1.03397f) {
                                    if (h_w <= 1.24797f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.98088f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.01686f) {
                                        if (h_w <= 1.25458f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.28220f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28907f) {
                            if (j_w <= 0.97182f) {
                                if (h_w <= 1.25736f) {
                                    if (f_j <= 0.87682f) {
                                        if (j_w <= 0.96888f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (h_w <= 1.25844f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.03283f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (j_w <= 0.95426f) {
                if (h_w <= 1.15380f) {
                    if (f_j <= 0.89066f) {
                        if (h_w <= 1.12912f) {
                            votes[4]++
                        } else {
                            if (c_j <= 1.04882f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (j_w <= 0.94918f) {
                        if (c_j <= 1.05535f) {
                            if (f_j <= 0.89170f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (h_w <= 1.26243f) {
                                if (f_j <= 0.90660f) {
                                    if (c_j <= 1.05629f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.94564f) {
                                        if (h_w <= 1.21733f) {
                                            if (f_j <= 0.92463f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.28582f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.92185f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.90717f) {
                            if (c_j <= 1.05119f) {
                                if (c_j <= 1.05063f) {
                                    if (c_j <= 1.04905f) {
                                        if (f_j <= 0.86996f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.23217f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.25670f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.15532f) {
                    if (f_j <= 0.86087f) {
                        if (h_w <= 1.14290f) {
                            votes[2]++
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (c_j <= 1.04589f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.04644f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.28773f) {
                        if (h_w <= 1.22181f) {
                            if (c_j <= 1.04599f) {
                                if (j_w <= 0.95813f) {
                                    if (f_j <= 0.92123f) {
                                        if (c_j <= 1.04463f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.96047f) {
                                        if (f_j <= 0.85734f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.87836f) {
                                                if (h_w <= 1.20295f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03999f) {
                                            if (h_w <= 1.18486f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.96329f) {
                                                    if (f_j <= 0.85715f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.18621f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.86882f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (j_w <= 0.96308f) {
                                if (j_w <= 0.95801f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (f_j <= 0.87014f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 27
        if (f_j <= 0.87658f) {
            if (h_w <= 1.12538f) {
                if (h_w <= 1.12092f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.12430f) {
                        votes[3]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (h_w <= 1.17362f) {
                    if (c_j <= 1.03747f) {
                        if (c_j <= 1.01951f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.02818f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.03056f) {
                                    if (f_j <= 0.85918f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.15617f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.15135f) {
                            if (c_j <= 1.03867f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.96047f) {
                                    votes[2]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (f_j <= 0.86220f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.83087f) {
                        if (c_j <= 1.03337f) {
                            if (f_j <= 0.81918f) {
                                if (f_j <= 0.81788f) {
                                    if (f_j <= 0.79731f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.01535f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.81735f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.20985f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (f_j <= 0.84934f) {
                            if (h_w <= 1.19563f) {
                                if (j_w <= 0.97314f) {
                                    if (h_w <= 1.18179f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.84097f) {
                                    if (c_j <= 1.02559f) {
                                        if (h_w <= 1.22670f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.23765f) {
                                                votes[3]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03607f) {
                                            if (f_j <= 0.83395f) {
                                                votes[4]++
                                            } else {
                                                if (h_w <= 1.20899f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.98385f) {
                                        if (c_j <= 1.02785f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.96882f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.27783f) {
                                if (j_w <= 0.96454f) {
                                    if (f_j <= 0.86450f) {
                                        if (h_w <= 1.21689f) {
                                            if (j_w <= 0.95140f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.86026f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.87473f) {
                                            if (c_j <= 1.04125f) {
                                                if (h_w <= 1.19514f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.87217f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.02603f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.85803f) {
                                            if (h_w <= 1.22505f) {
                                                votes[3]++
                                            } else {
                                                if (h_w <= 1.23883f) {
                                                    votes[2]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.17781f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.03552f) {
                                                    if (j_w <= 0.97037f) {
                                                        if (j_w <= 0.96923f) {
                                                            if (f_j <= 0.85906f) {
                                                                votes[0]++
                                                            } else {
                                                                if (h_w <= 1.19674f) {
                                                                    votes[3]++
                                                                } else {
                                                                    if (h_w <= 1.20954f) {
                                                                        votes[4]++
                                                                    } else {
                                                                        votes[3]++
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.96545f) {
                                                        votes[4]++
                                                    } else {
                                                        if (c_j <= 1.03576f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15515f) {
                if (j_w <= 0.96167f) {
                    if (h_w <= 1.11663f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.04192f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.13309f) {
                                if (c_j <= 1.04945f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.94901f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.92143f) {
                                    votes[2]++
                                } else {
                                    if (f_j <= 0.92493f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (h_w <= 1.26243f) {
                        if (c_j <= 1.05957f) {
                            if (c_j <= 1.05127f) {
                                if (h_w <= 1.22822f) {
                                    if (f_j <= 0.87904f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.15688f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.04307f) {
                                                if (c_j <= 1.03910f) {
                                                    if (c_j <= 1.03226f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (c_j <= 1.04617f) {
                                                    if (c_j <= 1.04481f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.95418f) {
                                                        votes[1]++
                                                    } else {
                                                        if (c_j <= 1.04787f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.89012f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.24906f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.89229f) {
                                    if (c_j <= 1.05484f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.05731f) {
                                        if (f_j <= 0.90837f) {
                                            if (c_j <= 1.05370f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (j_w <= 0.95917f) {
                            if (c_j <= 1.07409f) {
                                if (f_j <= 0.90920f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.94751f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 28
        if (h_w <= 1.15755f) {
            if (f_j <= 0.87388f) {
                if (h_w <= 1.12944f) {
                    if (f_j <= 0.85717f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                } else {
                    if (h_w <= 1.15731f) {
                        if (h_w <= 1.15124f) {
                            if (j_w <= 0.96277f) {
                                if (h_w <= 1.14708f) {
                                    votes[4]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (c_j <= 1.01825f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.03144f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            votes[3]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (c_j <= 1.03763f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.11055f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.13728f) {
                            if (f_j <= 0.90240f) {
                                if (j_w <= 0.95135f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (c_j <= 1.03843f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.92143f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.94441f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (j_w <= 0.95727f) {
                if (j_w <= 0.94900f) {
                    if (c_j <= 1.05613f) {
                        if (f_j <= 0.89493f) {
                            if (c_j <= 1.05477f) {
                                if (c_j <= 1.05439f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (h_w <= 1.25768f) {
                            if (f_j <= 0.93183f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (c_j <= 1.06846f) {
                                votes[4]++
                            } else {
                                if (h_w <= 1.27455f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.19365f) {
                        votes[0]++
                    } else {
                        if (j_w <= 0.95237f) {
                            if (h_w <= 1.26270f) {
                                if (j_w <= 0.95128f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (c_j <= 1.04655f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.04877f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.30735f) {
                    if (h_w <= 1.22135f) {
                        if (f_j <= 0.83853f) {
                            if (c_j <= 1.02490f) {
                                if (h_w <= 1.21012f) {
                                    if (f_j <= 0.81684f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (j_w <= 0.97144f) {
                                    if (j_w <= 0.96922f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.19167f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96417f) {
                                if (j_w <= 0.96308f) {
                                    if (f_j <= 0.84604f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.95965f) {
                                            if (h_w <= 1.20065f) {
                                                votes[4]++
                                            } else {
                                                if (h_w <= 1.20840f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.03871f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.85563f) {
                                                    votes[1]++
                                                } else {
                                                    if (j_w <= 0.96010f) {
                                                        votes[1]++
                                                    } else {
                                                        if (h_w <= 1.20814f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (c_j <= 1.02773f) {
                                    if (c_j <= 1.02653f) {
                                        if (f_j <= 0.84663f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.97883f) {
                                                votes[3]++
                                            } else {
                                                if (h_w <= 1.17228f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.97359f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.97390f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.15908f) {
                                        if (j_w <= 0.96869f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (j_w <= 0.96643f) {
                                            if (c_j <= 1.03505f) {
                                                votes[0]++
                                            } else {
                                                if (h_w <= 1.18916f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.86822f) {
                                                if (c_j <= 1.02830f) {
                                                    if (j_w <= 0.97259f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (c_j <= 1.03259f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03713f) {
                            if (f_j <= 0.86878f) {
                                if (f_j <= 0.82988f) {
                                    if (j_w <= 0.96715f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.24995f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.85211f) {
                                        if (f_j <= 0.83813f) {
                                            if (j_w <= 0.97458f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.97006f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.97247f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.24554f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96069f) {
                                if (f_j <= 0.86581f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 29
        if (h_w <= 1.12198f) {
            if (h_w <= 1.11008f) {
                votes[2]++
            } else {
                if (h_w <= 1.11011f) {
                    votes[1]++
                } else {
                    votes[2]++
                }
            }
        } else {
            if (j_w <= 0.95639f) {
                if (h_w <= 1.16425f) {
                    if (c_j <= 1.04649f) {
                        votes[0]++
                    } else {
                        if (j_w <= 0.95210f) {
                            if (h_w <= 1.12592f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.90321f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.05886f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    }
                } else {
                    if (c_j <= 1.04793f) {
                        if (f_j <= 0.89505f) {
                            if (j_w <= 0.95552f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (c_j <= 1.05374f) {
                            if (f_j <= 0.92039f) {
                                if (j_w <= 0.95130f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.22269f) {
                                        if (j_w <= 0.95400f) {
                                            if (c_j <= 1.04905f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.88504f) {
                                if (j_w <= 0.94807f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.20319f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.93592f) {
                                    if (h_w <= 1.27455f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (f_j <= 0.90286f) {
                                        if (j_w <= 0.94193f) {
                                            if (f_j <= 0.88875f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (h_w <= 1.21248f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.91750f) {
                                                if (f_j <= 0.90660f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f_j <= 0.85023f) {
                    if (h_w <= 1.19563f) {
                        if (j_w <= 0.97417f) {
                            if (f_j <= 0.83820f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.04081f) {
                                    if (h_w <= 1.15028f) {
                                        votes[2]++
                                    } else {
                                        if (c_j <= 1.03697f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97977f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    } else {
                        if (f_j <= 0.83033f) {
                            if (f_j <= 0.82096f) {
                                if (j_w <= 0.98942f) {
                                    if (f_j <= 0.80684f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.98490f) {
                                            if (h_w <= 1.24253f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (h_w <= 1.23290f) {
                                    if (c_j <= 1.01869f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.98385f) {
                                if (c_j <= 1.03632f) {
                                    if (f_j <= 0.83891f) {
                                        if (h_w <= 1.23005f) {
                                            votes[3]++
                                        } else {
                                            if (j_w <= 0.97458f) {
                                                votes[4]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.84464f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.97291f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03926f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.03579f) {
                        if (c_j <= 1.02639f) {
                            if (f_j <= 0.86450f) {
                                if (c_j <= 1.02449f) {
                                    if (c_j <= 1.02034f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.97541f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.89130f) {
                                if (h_w <= 1.24360f) {
                                    if (h_w <= 1.20954f) {
                                        if (j_w <= 0.96549f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.16097f) {
                                                if (j_w <= 0.97122f) {
                                                    if (c_j <= 1.03292f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (f_j <= 0.85918f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.19740f) {
                                                    if (c_j <= 1.02773f) {
                                                        if (c_j <= 1.02738f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.96705f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.19968f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96621f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03053f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.02921f) {
                                        if (j_w <= 0.97182f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03200f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.95730f) {
                            if (f_j <= 0.88631f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (h_w <= 1.28773f) {
                                if (c_j <= 1.04389f) {
                                    if (h_w <= 1.14183f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96501f) {
                                            if (h_w <= 1.23130f) {
                                                if (f_j <= 0.85909f) {
                                                    if (j_w <= 0.96279f) {
                                                        votes[4]++
                                                    } else {
                                                        if (j_w <= 0.96417f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.04064f) {
                                                        if (c_j <= 1.03871f) {
                                                            if (f_j <= 0.87948f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.96209f) {
                                                                if (f_j <= 0.88461f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                if (f_j <= 0.90727f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[2]++
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        if (c_j <= 1.04209f) {
                                                            votes[1]++
                                                        } else {
                                                            if (j_w <= 0.95883f) {
                                                                votes[1]++
                                                            } else {
                                                                if (f_j <= 0.86961f) {
                                                                    votes[2]++
                                                                } else {
                                                                    votes[4]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.04044f) {
                                                    if (h_w <= 1.26028f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            }
        }
    }

    private fun evaluateTrees30to39(
        h_w: Float,
        j_w: Float,
        f_j: Float,
        c_j: Float,
        votes: IntArray
    ) {
        // Tree 30
        if (h_w <= 1.15485f) {
            if (h_w <= 1.12217f) {
                if (h_w <= 1.11008f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.11011f) {
                        votes[1]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (c_j <= 1.04238f) {
                    if (f_j <= 0.89979f) {
                        if (c_j <= 1.02266f) {
                            votes[2]++
                        } else {
                            if (j_w <= 0.96427f) {
                                if (c_j <= 1.03878f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (j_w <= 0.95612f) {
                        if (h_w <= 1.13385f) {
                            if (c_j <= 1.04795f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.05373f) {
                                    if (c_j <= 1.05113f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            }
        } else {
            if (h_w <= 1.20408f) {
                if (c_j <= 1.03555f) {
                    if (j_w <= 0.97610f) {
                        if (j_w <= 0.97032f) {
                            if (h_w <= 1.19674f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.19935f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97088f) {
                                if (c_j <= 1.03041f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.87335f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.02639f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.97646f) {
                            votes[0]++
                        } else {
                            if (h_w <= 1.19280f) {
                                if (f_j <= 0.81934f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.86044f) {
                                        if (h_w <= 1.15755f) {
                                            if (f_j <= 0.83301f) {
                                                votes[3]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.15921f) {
                        if (h_w <= 1.15716f) {
                            votes[0]++
                        } else {
                            if (c_j <= 1.05139f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (f_j <= 0.91528f) {
                            if (h_w <= 1.20206f) {
                                if (f_j <= 0.89950f) {
                                    if (c_j <= 1.05005f) {
                                        if (f_j <= 0.83193f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.04645f) {
                                                if (h_w <= 1.18909f) {
                                                    if (j_w <= 0.95967f) {
                                                        votes[4]++
                                                    } else {
                                                        if (f_j <= 0.85762f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    if (h_w <= 1.19640f) {
                                                        if (c_j <= 1.03672f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.91951f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.18261f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.95122f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.30735f) {
                    if (f_j <= 0.85053f) {
                        if (j_w <= 0.98737f) {
                            if (f_j <= 0.83813f) {
                                if (c_j <= 1.02608f) {
                                    if (f_j <= 0.82422f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.96717f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.96496f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.98385f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.25458f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.00851f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.01048f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.04125f) {
                            if (j_w <= 0.96595f) {
                                if (f_j <= 0.87280f) {
                                    if (f_j <= 0.86383f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (f_j <= 0.88239f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97336f) {
                                    if (h_w <= 1.24483f) {
                                        if (f_j <= 0.85878f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (f_j <= 0.86740f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (j_w <= 0.94631f) {
                                if (c_j <= 1.05726f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.26243f) {
                                        if (f_j <= 0.88632f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (c_j <= 1.07409f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.91402f) {
                                    if (j_w <= 0.94920f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.22279f) {
                                            votes[1]++
                                        } else {
                                            if (f_j <= 0.90717f) {
                                                if (f_j <= 0.88678f) {
                                                    if (h_w <= 1.24613f) {
                                                        if (f_j <= 0.88317f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 31
        if (j_w <= 0.96703f) {
            if (j_w <= 0.94533f) {
                if (h_w <= 1.21733f) {
                    if (j_w <= 0.94078f) {
                        if (f_j <= 0.91637f) {
                            votes[2]++
                        } else {
                            votes[0]++
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (f_j <= 0.90319f) {
                        if (f_j <= 0.87851f) {
                            votes[1]++
                        } else {
                            votes[4]++
                        }
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (h_w <= 1.15408f) {
                    if (c_j <= 1.04589f) {
                        if (f_j <= 0.87209f) {
                            if (h_w <= 1.14167f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.15135f) {
                                    if (h_w <= 1.14290f) {
                                        votes[2]++
                                    } else {
                                        if (f_j <= 0.86087f) {
                                            votes[4]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96103f) {
                                if (h_w <= 1.14610f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    } else {
                        if (c_j <= 1.04713f) {
                            votes[0]++
                        } else {
                            if (h_w <= 1.12912f) {
                                if (f_j <= 0.88611f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.13931f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.95717f) {
                        if (j_w <= 0.95552f) {
                            if (c_j <= 1.04793f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.29909f) {
                                    if (c_j <= 1.05737f) {
                                        if (c_j <= 1.04877f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.87834f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.17039f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.89793f) {
                                                        if (f_j <= 0.88891f) {
                                                            votes[0]++
                                                        } else {
                                                            if (j_w <= 0.95067f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    } else {
                                                        if (f_j <= 0.90151f) {
                                                            votes[4]++
                                                        } else {
                                                            if (j_w <= 0.95237f) {
                                                                if (h_w <= 1.21248f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (h_w <= 1.24151f) {
                                                                        votes[3]++
                                                                    } else {
                                                                        votes[0]++
                                                                    }
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    } else {
                        if (h_w <= 1.28696f) {
                            if (c_j <= 1.03926f) {
                                if (c_j <= 1.03833f) {
                                    if (f_j <= 0.87280f) {
                                        if (h_w <= 1.19365f) {
                                            if (f_j <= 0.86395f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.86445f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.03508f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.22339f) {
                                            if (c_j <= 1.03658f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.20206f) {
                                    if (h_w <= 1.15892f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.96056f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.03956f) {
                                                votes[4]++
                                            } else {
                                                if (h_w <= 1.19179f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.04072f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.20853f) {
                                            if (h_w <= 1.20295f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.04100f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.95803f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.19106f) {
                if (j_w <= 0.97943f) {
                    if (f_j <= 0.86813f) {
                        if (f_j <= 0.82178f) {
                            if (j_w <= 0.97180f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.12816f) {
                                if (f_j <= 0.85711f) {
                                    votes[2]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (f_j <= 0.85774f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.85872f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.15665f) {
                            votes[3]++
                        } else {
                            if (j_w <= 0.97549f) {
                                if (f_j <= 0.87311f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.87740f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.98325f) {
                        if (h_w <= 1.15030f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.85646f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (c_j <= 1.00805f) {
                            if (f_j <= 0.82650f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[2]++
                        }
                    }
                }
            } else {
                if (f_j <= 0.84981f) {
                    if (f_j <= 0.83232f) {
                        if (j_w <= 0.98217f) {
                            if (j_w <= 0.96986f) {
                                if (c_j <= 1.03337f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.97719f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (j_w <= 0.98571f) {
                                votes[1]++
                            } else {
                                if (j_w <= 0.99108f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.24253f) {
                            if (c_j <= 1.01299f) {
                                votes[3]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (f_j <= 0.84560f) {
                                if (c_j <= 1.01536f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.28907f) {
                        if (h_w <= 1.22243f) {
                            if (f_j <= 0.86266f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.02946f) {
                                    votes[1]++
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (j_w <= 0.97182f) {
                                if (j_w <= 0.97016f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.24457f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.85465f) {
                                    votes[1]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 32
        if (h_w <= 1.14155f) {
            if (h_w <= 1.12217f) {
                if (j_w <= 0.97690f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.97971f) {
                        votes[1]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (c_j <= 1.04589f) {
                    if (f_j <= 0.88530f) {
                        if (h_w <= 1.13229f) {
                            if (c_j <= 1.02842f) {
                                if (j_w <= 0.97656f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.97987f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.01833f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    if (f_j <= 0.88611f) {
                        votes[4]++
                    } else {
                        votes[0]++
                    }
                }
            }
        } else {
            if (j_w <= 0.96454f) {
                if (h_w <= 1.29108f) {
                    if (c_j <= 1.03687f) {
                        votes[0]++
                    } else {
                        if (f_j <= 0.92504f) {
                            if (c_j <= 1.03727f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.27593f) {
                                    if (f_j <= 0.84267f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.20394f) {
                                            if (j_w <= 0.95325f) {
                                                if (j_w <= 0.94957f) {
                                                    if (f_j <= 0.90885f) {
                                                        votes[4]++
                                                    } else {
                                                        if (c_j <= 1.05776f) {
                                                            votes[0]++
                                                        } else {
                                                            if (f_j <= 0.91994f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (c_j <= 1.03936f) {
                                                    if (c_j <= 1.03871f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (j_w <= 0.96188f) {
                                                        if (h_w <= 1.19729f) {
                                                            if (h_w <= 1.19019f) {
                                                                if (j_w <= 0.95657f) {
                                                                    votes[2]++
                                                                } else {
                                                                    if (f_j <= 0.86087f) {
                                                                        votes[4]++
                                                                    } else {
                                                                        if (h_w <= 1.18187f) {
                                                                            if (c_j <= 1.04194f) {
                                                                                votes[1]++
                                                                            } else {
                                                                                votes[2]++
                                                                            }
                                                                        } else {
                                                                            votes[4]++
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        if (f_j <= 0.87501f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.94920f) {
                                                if (h_w <= 1.26185f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (h_w <= 1.22279f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.04044f) {
                                                        votes[4]++
                                                    } else {
                                                        if (f_j <= 0.90717f) {
                                                            if (c_j <= 1.05063f) {
                                                                if (j_w <= 0.95552f) {
                                                                    if (f_j <= 0.89317f) {
                                                                        votes[1]++
                                                                    } else {
                                                                        votes[0]++
                                                                    }
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            } else {
                                                                if (f_j <= 0.89413f) {
                                                                    votes[1]++
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (j_w <= 0.95007f) {
                                if (f_j <= 0.95380f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.92739f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.18855f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.18888f) {
                    if (c_j <= 1.02001f) {
                        if (f_j <= 0.83746f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.16265f) {
                                votes[2]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.17964f) {
                            if (j_w <= 0.97541f) {
                                if (j_w <= 0.97455f) {
                                    if (c_j <= 1.03059f) {
                                        if (c_j <= 1.02967f) {
                                            if (h_w <= 1.17401f) {
                                                if (j_w <= 0.97259f) {
                                                    if (j_w <= 0.97217f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (c_j <= 1.03031f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (f_j <= 0.88605f) {
                                if (f_j <= 0.84293f) {
                                    if (c_j <= 1.03011f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.30735f) {
                        if (f_j <= 0.84934f) {
                            if (c_j <= 1.01149f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.19490f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.96694f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.82988f) {
                                            if (h_w <= 1.20053f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.96986f) {
                                                    if (f_j <= 0.80343f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (f_j <= 0.81193f) {
                                                        votes[4]++
                                                    } else {
                                                        if (c_j <= 1.02455f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.01641f) {
                                                if (c_j <= 1.01536f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (f_j <= 0.83853f) {
                                                    if (f_j <= 0.83395f) {
                                                        votes[4]++
                                                    } else {
                                                        if (h_w <= 1.22316f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.19273f) {
                                if (c_j <= 1.03000f) {
                                    votes[0]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (j_w <= 0.96621f) {
                                    if (f_j <= 0.86364f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96596f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.21107f) {
                                        if (c_j <= 1.03274f) {
                                            votes[3]++
                                        } else {
                                            if (c_j <= 1.03383f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.03076f) {
                                            if (h_w <= 1.21574f) {
                                                votes[0]++
                                            } else {
                                                if (c_j <= 1.02679f) {
                                                    votes[2]++
                                                } else {
                                                    if (f_j <= 0.87096f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.85716f) {
                                                if (j_w <= 0.96957f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (f_j <= 0.87925f) {
                                                    if (f_j <= 0.86915f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 33
        if (h_w <= 1.15753f) {
            if (c_j <= 1.02163f) {
                if (h_w <= 1.12092f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.12430f) {
                        votes[3]++
                    } else {
                        if (j_w <= 0.98648f) {
                            votes[2]++
                        } else {
                            votes[3]++
                        }
                    }
                }
            } else {
                if (j_w <= 0.96732f) {
                    if (h_w <= 1.12243f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.12450f) {
                            votes[0]++
                        } else {
                            if (c_j <= 1.04960f) {
                                if (c_j <= 1.04139f) {
                                    if (h_w <= 1.15408f) {
                                        if (j_w <= 0.96216f) {
                                            votes[4]++
                                        } else {
                                            if (f_j <= 0.88018f) {
                                                if (h_w <= 1.14167f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (f_j <= 0.88528f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.92493f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.02337f) {
                        if (f_j <= 0.85111f) {
                            votes[3]++
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (h_w <= 1.14065f) {
                            if (j_w <= 0.96842f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[3]++
                        }
                    }
                }
            }
        } else {
            if (c_j <= 1.03470f) {
                if (h_w <= 1.30816f) {
                    if (h_w <= 1.19563f) {
                        if (f_j <= 0.87180f) {
                            if (h_w <= 1.17964f) {
                                votes[3]++
                            } else {
                                if (j_w <= 0.97191f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.19252f) {
                                        if (f_j <= 0.84057f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.18858f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.02639f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (f_j <= 0.83033f) {
                            if (h_w <= 1.20941f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.82099f) {
                                    if (h_w <= 1.24732f) {
                                        if (c_j <= 1.02628f) {
                                            if (f_j <= 0.80241f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (f_j <= 0.84934f) {
                                if (h_w <= 1.20142f) {
                                    if (f_j <= 0.84146f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (j_w <= 0.97270f) {
                                        if (j_w <= 0.96905f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.23950f) {
                                            if (j_w <= 0.97857f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (h_w <= 1.25873f) {
                                                if (j_w <= 0.98488f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.03263f) {
                                    if (c_j <= 1.02900f) {
                                        if (j_w <= 0.97478f) {
                                            if (c_j <= 1.02644f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.23301f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (h_w <= 1.28907f) {
                                            if (c_j <= 1.03139f) {
                                                if (f_j <= 0.86740f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.22843f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.30021f) {
                    if (f_j <= 0.86434f) {
                        if (j_w <= 0.96195f) {
                            if (h_w <= 1.21689f) {
                                if (f_j <= 0.83309f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.96168f) {
                                        if (j_w <= 0.96001f) {
                                            if (f_j <= 0.86098f) {
                                                if (h_w <= 1.19745f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (j_w <= 0.96425f) {
                                if (h_w <= 1.19841f) {
                                    votes[1]++
                                } else {
                                    if (c_j <= 1.03926f) {
                                        if (f_j <= 0.85213f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03533f) {
                                    if (f_j <= 0.84292f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.19752f) {
                            if (h_w <= 1.16741f) {
                                if (c_j <= 1.04438f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.90441f) {
                                        votes[2]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.94564f) {
                                    votes[2]++
                                } else {
                                    if (f_j <= 0.90958f) {
                                        if (h_w <= 1.17533f) {
                                            if (j_w <= 0.95727f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (c_j <= 1.04607f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.95011f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.20408f) {
                                if (j_w <= 0.95838f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.87204f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.88948f) {
                                    if (h_w <= 1.22688f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.04655f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.95138f) {
                                                if (f_j <= 0.88584f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95818f) {
                                        if (h_w <= 1.27019f) {
                                            if (c_j <= 1.05900f) {
                                                if (j_w <= 0.94710f) {
                                                    if (f_j <= 0.91783f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    if (f_j <= 0.90831f) {
                                                        votes[1]++
                                                    } else {
                                                        if (f_j <= 0.91630f) {
                                                            votes[4]++
                                                        } else {
                                                            if (h_w <= 1.24205f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (c_j <= 1.06926f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 34
        if (f_j <= 0.87885f) {
            if (h_w <= 1.13689f) {
                if (f_j <= 0.87500f) {
                    if (c_j <= 1.02165f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.97698f) {
                            votes[2]++
                        } else {
                            votes[1]++
                        }
                    }
                } else {
                    if (c_j <= 1.02393f) {
                        votes[3]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (j_w <= 0.96814f) {
                    if (h_w <= 1.22499f) {
                        if (j_w <= 0.95784f) {
                            if (j_w <= 0.95065f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (f_j <= 0.84267f) {
                                votes[0]++
                            } else {
                                if (j_w <= 0.96121f) {
                                    if (h_w <= 1.15572f) {
                                        if (j_w <= 0.95861f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        if (c_j <= 1.04269f) {
                                            if (h_w <= 1.20712f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.86535f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.18909f) {
                                        if (c_j <= 1.03651f) {
                                            if (h_w <= 1.17361f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (j_w <= 0.96540f) {
                                            if (j_w <= 0.96330f) {
                                                if (j_w <= 0.96173f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.03555f) {
                                                if (f_j <= 0.86415f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.19674f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03419f) {
                            if (h_w <= 1.23750f) {
                                if (c_j <= 1.03337f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (h_w <= 1.25859f) {
                                if (c_j <= 1.04125f) {
                                    votes[1]++
                                } else {
                                    if (c_j <= 1.04602f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.86629f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.96084f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.01149f) {
                        votes[0]++
                    } else {
                        if (h_w <= 1.24289f) {
                            if (h_w <= 1.17401f) {
                                if (f_j <= 0.86450f) {
                                    if (j_w <= 0.97103f) {
                                        if (f_j <= 0.86028f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (c_j <= 1.03072f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.83811f) {
                                    if (j_w <= 0.97077f) {
                                        if (f_j <= 0.81226f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (f_j <= 0.81193f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.01815f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.17560f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.03078f) {
                                            if (h_w <= 1.21197f) {
                                                if (j_w <= 0.97529f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.84693f) {
                                                        votes[4]++
                                                    } else {
                                                        if (f_j <= 0.85457f) {
                                                            if (h_w <= 1.18634f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (f_j <= 0.84694f) {
                                                    if (c_j <= 1.02190f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.02808f) {
                                                        votes[2]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.96888f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.01646f) {
                                if (f_j <= 0.84646f) {
                                    if (j_w <= 0.98488f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (f_j <= 0.85090f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (c_j <= 1.04443f) {
                if (h_w <= 1.15189f) {
                    if (c_j <= 1.03906f) {
                        votes[2]++
                    } else {
                        if (f_j <= 0.89453f) {
                            votes[3]++
                        } else {
                            if (c_j <= 1.04339f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.89500f) {
                        if (j_w <= 0.96286f) {
                            votes[1]++
                        } else {
                            if (j_w <= 0.97103f) {
                                if (h_w <= 1.32797f) {
                                    if (j_w <= 0.96830f) {
                                        if (j_w <= 0.96730f) {
                                            if (h_w <= 1.17386f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.89130f) {
                                                    votes[3]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            } else {
                if (h_w <= 1.15657f) {
                    if (f_j <= 0.88390f) {
                        votes[4]++
                    } else {
                        if (h_w <= 1.12001f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.89769f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.06262f) {
                        if (f_j <= 0.89077f) {
                            if (c_j <= 1.05478f) {
                                if (h_w <= 1.23022f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.23323f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.27950f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (j_w <= 0.94894f) {
                                if (f_j <= 0.92954f) {
                                    if (h_w <= 1.25196f) {
                                        if (j_w <= 0.94661f) {
                                            if (j_w <= 0.94420f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (j_w <= 0.95237f) {
                                    if (c_j <= 1.05299f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.95514f) {
                                        if (h_w <= 1.23380f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.95730f) {
                                            if (h_w <= 1.18136f) {
                                                votes[2]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.06721f) {
                            votes[0]++
                        } else {
                            if (c_j <= 1.07576f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            }
        }
        // Tree 35
        if (j_w <= 0.95720f) {
            if (f_j <= 0.90236f) {
                if (c_j <= 1.04802f) {
                    if (j_w <= 0.95523f) {
                        if (c_j <= 1.04787f) {
                            votes[1]++
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (h_w <= 1.16639f) {
                            votes[2]++
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (j_w <= 0.94799f) {
                        if (c_j <= 1.05541f) {
                            votes[1]++
                        } else {
                            if (j_w <= 0.94444f) {
                                if (j_w <= 0.93923f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.06197f) {
                                        votes[2]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (h_w <= 1.17688f) {
                            if (h_w <= 1.12592f) {
                                votes[0]++
                            } else {
                                if (j_w <= 0.95135f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (f_j <= 0.88333f) {
                                if (h_w <= 1.24613f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.93637f) {
                    if (c_j <= 1.07576f) {
                        votes[1]++
                    } else {
                        votes[0]++
                    }
                } else {
                    if (h_w <= 1.14952f) {
                        if (j_w <= 0.95530f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.04589f) {
                                votes[2]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (h_w <= 1.20577f) {
                            if (h_w <= 1.15305f) {
                                votes[3]++
                            } else {
                                if (f_j <= 0.92484f) {
                                    if (j_w <= 0.95144f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (h_w <= 1.26436f) {
                                if (f_j <= 0.91630f) {
                                    if (j_w <= 0.94758f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                }
            }
        } else {
            if (j_w <= 0.98114f) {
                if (h_w <= 1.28501f) {
                    if (h_w <= 1.15739f) {
                        if (f_j <= 0.87388f) {
                            if (h_w <= 1.12816f) {
                                if (j_w <= 0.97698f) {
                                    votes[2]++
                                } else {
                                    if (h_w <= 1.10739f) {
                                        votes[2]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03747f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.86664f) {
                                        if (h_w <= 1.14290f) {
                                            votes[2]++
                                        } else {
                                            if (f_j <= 0.86087f) {
                                                votes[4]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.12254f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.13723f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03936f) {
                            if (f_j <= 0.84803f) {
                                if (f_j <= 0.83853f) {
                                    if (j_w <= 0.97513f) {
                                        if (f_j <= 0.83395f) {
                                            if (j_w <= 0.97357f) {
                                                if (h_w <= 1.23033f) {
                                                    if (h_w <= 1.21710f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.03316f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (c_j <= 1.03038f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.83638f) {
                                            if (c_j <= 1.02244f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.97563f) {
                                        if (h_w <= 1.20417f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.97505f) {
                                    if (j_w <= 0.97086f) {
                                        if (j_w <= 0.97048f) {
                                            if (c_j <= 1.03576f) {
                                                if (h_w <= 1.16097f) {
                                                    votes[3]++
                                                } else {
                                                    if (j_w <= 0.96879f) {
                                                        if (c_j <= 1.03390f) {
                                                            if (j_w <= 0.96832f) {
                                                                if (h_w <= 1.18118f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[4]++
                                                                }
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.96621f) {
                                                                if (f_j <= 0.86966f) {
                                                                    votes[0]++
                                                                } else {
                                                                    votes[3]++
                                                                }
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        if (h_w <= 1.24601f) {
                                                            votes[0]++
                                                        } else {
                                                            if (c_j <= 1.03172f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.03617f) {
                                                    if (h_w <= 1.20029f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.03676f) {
                                                        votes[3]++
                                                    } else {
                                                        if (f_j <= 0.87280f) {
                                                            votes[1]++
                                                        } else {
                                                            if (c_j <= 1.03871f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (c_j <= 1.02628f) {
                                            if (c_j <= 1.02603f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            if (h_w <= 1.23126f) {
                                                if (h_w <= 1.17545f) {
                                                    if (f_j <= 0.86775f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.18735f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.96192f) {
                                if (c_j <= 1.04202f) {
                                    if (j_w <= 0.96176f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.04204f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.19399f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.95738f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.95806f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.16696f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.99341f) {
                        if (c_j <= 1.01069f) {
                            votes[3]++
                        } else {
                            if (c_j <= 1.01686f) {
                                if (j_w <= 0.98488f) {
                                    if (c_j <= 1.01641f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.18811f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.82422f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.83718f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        votes[0]++
                    }
                }
            }
        }
        // Tree 36
        if (h_w <= 1.12198f) {
            if (j_w <= 0.97690f) {
                votes[2]++
            } else {
                if (j_w <= 0.97971f) {
                    votes[1]++
                } else {
                    votes[2]++
                }
            }
        } else {
            if (c_j <= 1.03454f) {
                if (h_w <= 1.19563f) {
                    if (f_j <= 0.87184f) {
                        if (j_w <= 0.97928f) {
                            if (h_w <= 1.18571f) {
                                if (h_w <= 1.16259f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.03023f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.85894f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.84890f) {
                                    if (c_j <= 1.02281f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (c_j <= 1.01746f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    } else {
                        if (j_w <= 0.97429f) {
                            if (c_j <= 1.02886f) {
                                votes[4]++
                            } else {
                                if (h_w <= 1.14800f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.03125f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.02276f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.20273f) {
                        if (h_w <= 1.19740f) {
                            if (h_w <= 1.19663f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (f_j <= 0.84995f) {
                            if (f_j <= 0.83033f) {
                                if (c_j <= 1.03337f) {
                                    if (h_w <= 1.24995f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.26766f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (h_w <= 1.21322f) {
                                    if (j_w <= 0.97250f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (h_w <= 1.24841f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.01641f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.85553f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.03308f) {
                                    votes[1]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (c_j <= 1.04474f) {
                    if (h_w <= 1.28696f) {
                        if (c_j <= 1.03497f) {
                            if (h_w <= 1.22098f) {
                                votes[0]++
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.86785f) {
                                if (f_j <= 0.86445f) {
                                    if (c_j <= 1.04265f) {
                                        if (f_j <= 0.84267f) {
                                            if (j_w <= 0.96307f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            if (h_w <= 1.17167f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.85302f) {
                                                    if (f_j <= 0.84604f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    if (c_j <= 1.03626f) {
                                                        votes[4]++
                                                    } else {
                                                        if (j_w <= 0.96425f) {
                                                            if (f_j <= 0.86395f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.03853f) {
                                        if (c_j <= 1.03576f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.96463f) {
                                    if (h_w <= 1.15201f) {
                                        votes[3]++
                                    } else {
                                        if (h_w <= 1.15521f) {
                                            votes[2]++
                                        } else {
                                            if (h_w <= 1.21042f) {
                                                if (j_w <= 0.96000f) {
                                                    if (c_j <= 1.04212f) {
                                                        votes[3]++
                                                    } else {
                                                        if (c_j <= 1.04460f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.03940f) {
                                                        if (h_w <= 1.19542f) {
                                                            if (f_j <= 0.90569f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            } else {
                                                if (f_j <= 0.88239f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (j_w <= 0.94847f) {
                        if (h_w <= 1.15305f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.92653f) {
                                if (h_w <= 1.26185f) {
                                    if (h_w <= 1.20612f) {
                                        if (j_w <= 0.94661f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.88661f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.22448f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.16102f) {
                            if (h_w <= 1.13048f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (h_w <= 1.25704f) {
                                if (h_w <= 1.17358f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.89994f) {
                                        if (c_j <= 1.05374f) {
                                            if (c_j <= 1.05030f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.95130f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (j_w <= 0.95363f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.95181f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                }
            }
        }
        // Tree 37
        if (h_w <= 1.15753f) {
            if (h_w <= 1.12243f) {
                if (c_j <= 1.02365f) {
                    if (f_j <= 0.85250f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.97880f) {
                            votes[1]++
                        } else {
                            votes[2]++
                        }
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (c_j <= 1.03167f) {
                    if (c_j <= 1.02042f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.12816f) {
                            votes[2]++
                        } else {
                            votes[3]++
                        }
                    }
                } else {
                    if (j_w <= 0.95139f) {
                        if (c_j <= 1.05373f) {
                            votes[4]++
                        } else {
                            votes[0]++
                        }
                    } else {
                        if (c_j <= 1.04589f) {
                            if (h_w <= 1.13516f) {
                                if (c_j <= 1.04460f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (j_w <= 0.96047f) {
                                    votes[2]++
                                } else {
                                    if (f_j <= 0.88189f) {
                                        votes[4]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.91042f) {
                                votes[2]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.31061f) {
                if (c_j <= 1.03676f) {
                    if (f_j <= 0.84359f) {
                        if (f_j <= 0.83852f) {
                            if (c_j <= 1.02371f) {
                                if (h_w <= 1.18409f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.81115f) {
                                        if (j_w <= 0.98942f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.22718f) {
                                            if (f_j <= 0.82769f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.21710f) {
                                    votes[0]++
                                } else {
                                    if (c_j <= 1.03316f) {
                                        if (f_j <= 0.83546f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (c_j <= 1.03397f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.02757f) {
                                if (h_w <= 1.19254f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (h_w <= 1.23304f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.15901f) {
                            votes[1]++
                        } else {
                            if (h_w <= 1.24360f) {
                                if (c_j <= 1.02628f) {
                                    if (f_j <= 0.86009f) {
                                        if (f_j <= 0.84862f) {
                                            if (f_j <= 0.84663f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (j_w <= 0.97463f) {
                                                votes[2]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (j_w <= 0.96549f) {
                                        if (h_w <= 1.21004f) {
                                            if (j_w <= 0.96545f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.86590f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.21107f) {
                                            if (f_j <= 0.86818f) {
                                                votes[3]++
                                            } else {
                                                if (h_w <= 1.18315f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.87137f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.87116f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.88123f) {
                                    if (j_w <= 0.97103f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.85175f) {
                                            if (f_j <= 0.84646f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.85941f) {
                        if (f_j <= 0.84604f) {
                            votes[4]++
                        } else {
                            if (c_j <= 1.04013f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (c_j <= 1.04793f) {
                            if (f_j <= 0.86581f) {
                                if (j_w <= 0.96212f) {
                                    votes[0]++
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (j_w <= 0.95963f) {
                                    if (c_j <= 1.04655f) {
                                        if (h_w <= 1.20065f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.95778f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (h_w <= 1.23325f) {
                                        if (h_w <= 1.21042f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.96062f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96322f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.88630f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.89413f) {
                                    if (c_j <= 1.06120f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (c_j <= 1.05299f) {
                                        if (h_w <= 1.19602f) {
                                            votes[1]++
                                        } else {
                                            if (j_w <= 0.95237f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.95352f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.90286f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.22795f) {
                                                if (f_j <= 0.92786f) {
                                                    if (f_j <= 0.91015f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[2]++
                                                }
                                            } else {
                                                if (j_w <= 0.92960f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.27019f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                votes[1]++
            }
        }

        // Tree 38
        if (h_w <= 1.14139f) {
            if (h_w <= 1.12243f) {
                if (c_j <= 1.02365f) {
                    if (j_w <= 0.98061f) {
                        votes[1]++
                    } else {
                        votes[2]++
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (j_w <= 0.95139f) {
                    if (f_j <= 0.88611f) {
                        votes[4]++
                    } else {
                        votes[0]++
                    }
                } else {
                    if (h_w <= 1.13372f) {
                        if (h_w <= 1.12920f) {
                            if (c_j <= 1.04589f) {
                                if (f_j <= 0.85068f) {
                                    if (j_w <= 0.97614f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            votes[3]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            }
        } else {
            if (j_w <= 0.96549f) {
                if (h_w <= 1.30088f) {
                    if (h_w <= 1.17735f) {
                        if (f_j <= 0.90311f) {
                            if (h_w <= 1.15779f) {
                                if (j_w <= 0.95934f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.96174f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.87260f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.17376f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.95727f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.15275f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.16474f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.17225f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28582f) {
                            if (j_w <= 0.95767f) {
                                if (c_j <= 1.04560f) {
                                    if (h_w <= 1.21430f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.94631f) {
                                        if (j_w <= 0.94193f) {
                                            if (h_w <= 1.24397f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.88875f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.05726f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.92954f) {
                                                    votes[4]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.89994f) {
                                            if (c_j <= 1.05374f) {
                                                if (c_j <= 1.05111f) {
                                                    if (h_w <= 1.23022f) {
                                                        if (h_w <= 1.19618f) {
                                                            votes[0]++
                                                        } else {
                                                            if (h_w <= 1.21420f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            if (j_w <= 0.94806f) {
                                                votes[0]++
                                            } else {
                                                if (c_j <= 1.04877f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.25670f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.84267f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.87473f) {
                                        if (f_j <= 0.86445f) {
                                            if (j_w <= 0.96490f) {
                                                if (f_j <= 0.84604f) {
                                                    votes[4]++
                                                } else {
                                                    if (f_j <= 0.85809f) {
                                                        votes[1]++
                                                    } else {
                                                        if (c_j <= 1.04194f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (f_j <= 0.86581f) {
                                                votes[0]++
                                            } else {
                                                if (h_w <= 1.19712f) {
                                                    votes[3]++
                                                } else {
                                                    if (c_j <= 1.03629f) {
                                                        votes[1]++
                                                    } else {
                                                        if (f_j <= 0.86993f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96239f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (f_j <= 0.84934f) {
                    if (c_j <= 1.01149f) {
                        if (h_w <= 1.22983f) {
                            if (f_j <= 0.80587f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            votes[0]++
                        }
                    } else {
                        if (h_w <= 1.19563f) {
                            if (c_j <= 1.02652f) {
                                if (c_j <= 1.01951f) {
                                    if (f_j <= 0.83746f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (c_j <= 1.02891f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96790f) {
                                if (f_j <= 0.81472f) {
                                    votes[1]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.24253f) {
                                    if (f_j <= 0.82824f) {
                                        if (j_w <= 0.98490f) {
                                            if (j_w <= 0.97609f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.97270f) {
                                            if (f_j <= 0.83680f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.84560f) {
                                        if (f_j <= 0.84112f) {
                                            if (j_w <= 0.98101f) {
                                                if (c_j <= 1.02728f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.22243f) {
                        if (f_j <= 0.86833f) {
                            if (c_j <= 1.02163f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.15950f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.97412f) {
                                        if (f_j <= 0.85744f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.85981f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.96941f) {
                                if (f_j <= 0.87296f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (c_j <= 1.02276f) {
                                    votes[3]++
                                } else {
                                    if (j_w <= 0.97429f) {
                                        if (c_j <= 1.02900f) {
                                            if (h_w <= 1.16509f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.28501f) {
                            if (c_j <= 1.02900f) {
                                if (c_j <= 1.02758f) {
                                    votes[2]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.96649f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.86167f) {
                                        if (f_j <= 0.85380f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 39
        if (h_w <= 1.14897f) {
            if (h_w <= 1.12040f) {
                votes[2]++
            } else {
                if (c_j <= 1.05031f) {
                    if (h_w <= 1.13516f) {
                        if (h_w <= 1.12709f) {
                            votes[0]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (f_j <= 0.86995f) {
                            if (h_w <= 1.14290f) {
                                if (f_j <= 0.84567f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            votes[2]++
                        }
                    }
                } else {
                    votes[4]++
                }
            }
        } else {
            if (f_j <= 0.85697f) {
                if (f_j <= 0.85023f) {
                    if (f_j <= 0.83033f) {
                        if (h_w <= 1.17314f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.81852f) {
                                if (c_j <= 1.01535f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.81753f) {
                                        if (c_j <= 1.03280f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.02831f) {
                                    if (h_w <= 1.20476f) {
                                        if (j_w <= 0.97571f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (h_w <= 1.19472f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.03162f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.84934f) {
                            if (h_w <= 1.19648f) {
                                if (j_w <= 0.97389f) {
                                    votes[0]++
                                } else {
                                    if (c_j <= 1.01990f) {
                                        if (f_j <= 0.83746f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.96592f) {
                                    if (j_w <= 0.96223f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.84451f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.21146f) {
                                        if (j_w <= 0.97378f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (c_j <= 1.03788f) {
                        if (h_w <= 1.22068f) {
                            votes[3]++
                        } else {
                            if (c_j <= 1.02758f) {
                                votes[2]++
                            } else {
                                if (f_j <= 0.85308f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    } else {
                        votes[4]++
                    }
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (c_j <= 1.03576f) {
                        if (h_w <= 1.20866f) {
                            if (f_j <= 0.86813f) {
                                if (f_j <= 0.85872f) {
                                    votes[0]++
                                } else {
                                    if (j_w <= 0.96599f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.97945f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.88190f) {
                                    if (h_w <= 1.16671f) {
                                        if (c_j <= 1.02900f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (f_j <= 0.87571f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (j_w <= 0.96857f) {
                                if (j_w <= 0.96730f) {
                                    if (f_j <= 0.86666f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.88305f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (h_w <= 1.15515f) {
                            if (h_w <= 1.15201f) {
                                if (c_j <= 1.04317f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (c_j <= 1.04474f) {
                                if (f_j <= 0.92739f) {
                                    if (h_w <= 1.15716f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.95813f) {
                                            if (h_w <= 1.20521f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (j_w <= 0.95905f) {
                                                votes[1]++
                                            } else {
                                                if (h_w <= 1.18621f) {
                                                    if (j_w <= 0.96209f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (f_j <= 0.85809f) {
                                                        votes[1]++
                                                    } else {
                                                        if (j_w <= 0.96087f) {
                                                            if (f_j <= 0.87404f) {
                                                                if (h_w <= 1.23130f) {
                                                                    if (j_w <= 0.96062f) {
                                                                        votes[1]++
                                                                    } else {
                                                                        votes[0]++
                                                                    }
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.96545f) {
                                                                if (h_w <= 1.18938f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (c_j <= 1.03982f) {
                                                                        votes[4]++
                                                                    } else {
                                                                        if (f_j <= 0.86116f) {
                                                                            votes[4]++
                                                                        } else {
                                                                            votes[3]++
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.88437f) {
                                    if (c_j <= 1.05374f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.05439f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.89413f) {
                                        if (c_j <= 1.06471f) {
                                            if (j_w <= 0.95523f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.04640f) {
                                                    votes[0]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.89768f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.06222f) {
                                                if (f_j <= 0.91402f) {
                                                    if (c_j <= 1.04869f) {
                                                        votes[1]++
                                                    } else {
                                                        if (j_w <= 0.94475f) {
                                                            if (j_w <= 0.94231f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.05737f) {
                                                        if (h_w <= 1.24205f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.27455f) {
                                                    if (j_w <= 0.93726f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
    }

    private fun evaluateTrees40to49(
        h_w: Float,
        j_w: Float,
        f_j: Float,
        c_j: Float,
        votes: IntArray
    ) {
        // Tree 40
        if (j_w <= 0.97089f) {
            if (f_j <= 0.89196f) {
                if (h_w <= 1.28937f) {
                    if (h_w <= 1.12228f) {
                        votes[2]++
                    } else {
                        if (f_j <= 0.89044f) {
                            if (h_w <= 1.25919f) {
                                if (j_w <= 0.94968f) {
                                    if (f_j <= 0.88943f) {
                                        votes[0]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (c_j <= 1.03172f) {
                                        if (j_w <= 0.97048f) {
                                            if (f_j <= 0.85695f) {
                                                if (c_j <= 1.03139f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (f_j <= 0.88379f) {
                                            if (f_j <= 0.83022f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.04097f) {
                                                    if (j_w <= 0.96187f) {
                                                        votes[4]++
                                                    } else {
                                                        if (c_j <= 1.03398f) {
                                                            if (c_j <= 1.03250f) {
                                                                votes[3]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            if (f_j <= 0.87171f) {
                                                                if (f_j <= 0.86544f) {
                                                                    if (c_j <= 1.03518f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        if (j_w <= 0.96195f) {
                                                                            votes[1]++
                                                                        } else {
                                                                            if (h_w <= 1.17465f) {
                                                                                if (f_j <= 0.85114f) {
                                                                                    votes[2]++
                                                                                } else {
                                                                                    votes[3]++
                                                                                }
                                                                            } else {
                                                                                if (h_w <= 1.20912f) {
                                                                                    if (h_w <= 1.19365f) {
                                                                                        votes[4]++
                                                                                    } else {
                                                                                        if (f_j <= 0.85140f) {
                                                                                            votes[4]++
                                                                                        } else {
                                                                                            votes[1]++
                                                                                        }
                                                                                    }
                                                                                } else {
                                                                                    if (j_w <= 0.96392f) {
                                                                                        votes[0]++
                                                                                    } else {
                                                                                        votes[3]++
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    if (j_w <= 0.95967f) {
                                                        if (h_w <= 1.17605f) {
                                                            votes[4]++
                                                        } else {
                                                            if (h_w <= 1.21417f) {
                                                                if (h_w <= 1.19515f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (c_j <= 1.04212f) {
                                                                        votes[3]++
                                                                    } else {
                                                                        votes[4]++
                                                                    }
                                                                }
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.03956f) {
                                                if (j_w <= 0.96414f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (j_w <= 0.95390f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            votes[4]++
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (c_j <= 1.04894f) {
                    if (h_w <= 1.16127f) {
                        if (f_j <= 0.91026f) {
                            if (j_w <= 0.96220f) {
                                if (c_j <= 1.04261f) {
                                    if (j_w <= 0.96075f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    if (f_j <= 0.89679f) {
                                        votes[2]++
                                    } else {
                                        if (j_w <= 0.95600f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (h_w <= 1.13517f) {
                                votes[0]++
                            } else {
                                votes[2]++
                            }
                        }
                    } else {
                        if (f_j <= 0.89577f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.95764f) {
                                if (c_j <= 1.04760f) {
                                    votes[1]++
                                } else {
                                    if (j_w <= 0.95418f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.19944f) {
                                    if (h_w <= 1.18007f) {
                                        if (h_w <= 1.16907f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.05783f) {
                        if (c_j <= 1.05613f) {
                            if (c_j <= 1.05381f) {
                                if (h_w <= 1.18284f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.95220f) {
                                        if (h_w <= 1.26150f) {
                                            if (f_j <= 0.89413f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (h_w <= 1.26436f) {
                                votes[0]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (j_w <= 0.94078f) {
                            if (f_j <= 0.91637f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.20964f) {
                                    votes[0]++
                                } else {
                                    if (c_j <= 1.07518f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.94201f) {
                                votes[2]++
                            } else {
                                if (h_w <= 1.21733f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.06122f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (j_w <= 0.98114f) {
                if (h_w <= 1.19913f) {
                    if (c_j <= 1.01972f) {
                        votes[1]++
                    } else {
                        if (h_w <= 1.13521f) {
                            if (c_j <= 1.02365f) {
                                votes[1]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (f_j <= 0.89156f) {
                                if (h_w <= 1.18137f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.82093f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.19245f) {
                                            if (h_w <= 1.19118f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (h_w <= 1.19563f) {
                                                votes[3]++
                                            } else {
                                                if (j_w <= 0.97451f) {
                                                    votes[3]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84849f) {
                        if (j_w <= 0.97916f) {
                            votes[4]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (h_w <= 1.23883f) {
                            if (j_w <= 0.97321f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.02603f) {
                                    votes[0]++
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (f_j <= 0.88277f) {
                                votes[1]++
                            } else {
                                if (h_w <= 1.30290f) {
                                    votes[0]++
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.15769f) {
                    if (f_j <= 0.86407f) {
                        votes[2]++
                    } else {
                        votes[3]++
                    }
                } else {
                    if (j_w <= 0.98864f) {
                        if (h_w <= 1.19166f) {
                            votes[3]++
                        } else {
                            if (f_j <= 0.85608f) {
                                if (c_j <= 1.01641f) {
                                    if (f_j <= 0.84112f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (h_w <= 1.22983f) {
                            if (f_j <= 0.80587f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.21012f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    }
                }
            }
        }
        // Tree 41
        if (h_w <= 1.15485f) {
            if (h_w <= 1.12243f) {
                if (h_w <= 1.11008f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.02572f) {
                        if (c_j <= 1.01978f) {
                            votes[2]++
                        } else {
                            votes[1]++
                        }
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (c_j <= 1.03775f) {
                    if (f_j <= 0.83665f) {
                        votes[2]++
                    } else {
                        if (c_j <= 1.02505f) {
                            votes[3]++
                        } else {
                            if (j_w <= 0.97496f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.89208f) {
                        if (c_j <= 1.04317f) {
                            if (j_w <= 0.96047f) {
                                votes[2]++
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (c_j <= 1.04817f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.05373f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.12433f) {
                            votes[0]++
                        } else {
                            votes[2]++
                        }
                    }
                }
            }
        } else {
            if (f_j <= 0.87369f) {
                if (h_w <= 1.19490f) {
                    if (f_j <= 0.87311f) {
                        if (f_j <= 0.81754f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.96558f) {
                                if (f_j <= 0.86445f) {
                                    if (f_j <= 0.85239f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (f_j <= 0.86489f) {
                                        votes[0]++
                                    } else {
                                        if (c_j <= 1.04292f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.87124f) {
                                    if (j_w <= 0.98059f) {
                                        if (h_w <= 1.16354f) {
                                            if (j_w <= 0.97283f) {
                                                if (c_j <= 1.03059f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (c_j <= 1.01911f) {
                                            votes[2]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        }
                    } else {
                        votes[4]++
                    }
                } else {
                    if (j_w <= 0.95944f) {
                        votes[1]++
                    } else {
                        if (f_j <= 0.82502f) {
                            if (f_j <= 0.80343f) {
                                if (h_w <= 1.22720f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (h_w <= 1.24995f) {
                                    if (c_j <= 1.01535f) {
                                        if (h_w <= 1.20999f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (h_w <= 1.30735f) {
                                if (j_w <= 0.96717f) {
                                    if (c_j <= 1.03831f) {
                                        if (h_w <= 1.25536f) {
                                            if (j_w <= 0.96611f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.84292f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.96121f) {
                                            if (f_j <= 0.86186f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (j_w <= 0.96196f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.96830f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.97193f) {
                                            if (f_j <= 0.85874f) {
                                                votes[1]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.01064f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.02538f) {
                                                    if (f_j <= 0.83813f) {
                                                        votes[1]++
                                                    } else {
                                                        if (c_j <= 1.01641f) {
                                                            if (c_j <= 1.01536f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    if (j_w <= 0.97419f) {
                                                        if (f_j <= 0.84573f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    } else {
                                                        if (c_j <= 1.02603f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[2]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.29114f) {
                    if (f_j <= 0.89630f) {
                        if (c_j <= 1.02751f) {
                            if (f_j <= 0.88077f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (c_j <= 1.03872f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.04366f) {
                                    if (h_w <= 1.20814f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (f_j <= 0.88064f) {
                                        if (c_j <= 1.05021f) {
                                            if (f_j <= 0.87814f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (h_w <= 1.19375f) {
                                            votes[2]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.07576f) {
                            if (j_w <= 0.93776f) {
                                votes[1]++
                            } else {
                                if (f_j <= 0.90050f) {
                                    if (c_j <= 1.05374f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (c_j <= 1.05550f) {
                                        if (f_j <= 0.91528f) {
                                            if (f_j <= 0.91430f) {
                                                if (j_w <= 0.95363f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.18120f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.19988f) {
                                                            if (f_j <= 0.90495f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            if (h_w <= 1.19928f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.95189f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.21067f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.94631f) {
                                            if (c_j <= 1.05957f) {
                                                if (j_w <= 0.94533f) {
                                                    votes[2]++
                                                } else {
                                                    if (c_j <= 1.05726f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.19975f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 42
        if (h_w <= 1.15755f) {
            if (h_w <= 1.12217f) {
                if (c_j <= 1.02799f) {
                    if (f_j <= 0.85250f) {
                        votes[2]++
                    } else {
                        votes[1]++
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (h_w <= 1.12410f) {
                    votes[0]++
                } else {
                    if (j_w <= 0.95275f) {
                        if (f_j <= 0.88528f) {
                            votes[4]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (f_j <= 0.89979f) {
                            if (j_w <= 0.97928f) {
                                if (h_w <= 1.15135f) {
                                    if (j_w <= 0.95934f) {
                                        votes[2]++
                                    } else {
                                        if (f_j <= 0.85301f) {
                                            if (h_w <= 1.14299f) {
                                                if (h_w <= 1.12690f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (f_j <= 0.87342f) {
                                                if (j_w <= 0.96561f) {
                                                    votes[4]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03671f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03888f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            votes[2]++
                        }
                    }
                }
            }
        } else {
            if (j_w <= 0.96454f) {
                if (h_w <= 1.30021f) {
                    if (f_j <= 0.92690f) {
                        if (h_w <= 1.24661f) {
                            if (c_j <= 1.05613f) {
                                if (h_w <= 1.20206f) {
                                    if (c_j <= 1.04156f) {
                                        if (h_w <= 1.20078f) {
                                            if (j_w <= 0.96169f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.03822f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.03921f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (h_w <= 1.17376f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.95727f) {
                                                if (h_w <= 1.17739f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.04841f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.21423f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.23855f) {
                                            if (j_w <= 0.96038f) {
                                                if (f_j <= 0.88678f) {
                                                    if (c_j <= 1.04417f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                if (f_j <= 0.87882f) {
                                                    votes[1]++
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (f_j <= 0.88099f) {
                                if (f_j <= 0.86876f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (h_w <= 1.27077f) {
                                    if (f_j <= 0.91648f) {
                                        if (f_j <= 0.90717f) {
                                            if (f_j <= 0.89989f) {
                                                votes[4]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.95511f) {
                            if (j_w <= 0.95606f) {
                                if (h_w <= 1.25826f) {
                                    if (h_w <= 1.21412f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (f_j <= 0.87885f) {
                    if (h_w <= 1.22244f) {
                        if (h_w <= 1.19118f) {
                            if (c_j <= 1.03467f) {
                                if (j_w <= 0.97484f) {
                                    if (c_j <= 1.02773f) {
                                        if (c_j <= 1.02699f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.97248f) {
                                            if (f_j <= 0.85846f) {
                                                if (h_w <= 1.16951f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (j_w <= 0.97264f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.20273f) {
                                if (f_j <= 0.83862f) {
                                    if (f_j <= 0.82918f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.97501f) {
                                        if (j_w <= 0.96567f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.96830f) {
                                                if (c_j <= 1.03414f) {
                                                    votes[4]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.19409f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.84995f) {
                                    if (f_j <= 0.83163f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.97439f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03237f) {
                                        votes[3]++
                                    } else {
                                        if (j_w <= 0.96620f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.85175f) {
                            if (c_j <= 1.01279f) {
                                votes[0]++
                            } else {
                                if (f_j <= 0.83852f) {
                                    if (j_w <= 0.97458f) {
                                        if (c_j <= 1.03316f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.03397f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.82422f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.01910f) {
                                                votes[1]++
                                            } else {
                                                if (c_j <= 1.02267f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (h_w <= 1.24071f) {
                                if (f_j <= 0.85699f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.03076f) {
                                        votes[3]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.28907f) {
                        if (j_w <= 0.96621f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.22844f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                }
            }
        }
        // Tree 43
        if (h_w <= 1.12243f) {
            votes[2]++
        } else {
            if (j_w <= 0.96441f) {
                if (h_w <= 1.29917f) {
                    if (h_w <= 1.15628f) {
                        if (j_w <= 0.95139f) {
                            if (f_j <= 0.90321f) {
                                votes[4]++
                            } else {
                                if (h_w <= 1.15304f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            }
                        } else {
                            if (f_j <= 0.91042f) {
                                if (h_w <= 1.13728f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.04116f) {
                                        if (f_j <= 0.84841f) {
                                            votes[2]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.86842f) {
                                            votes[2]++
                                        } else {
                                            if (h_w <= 1.15031f) {
                                                votes[2]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.95530f) {
                                    votes[2]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.23639f) {
                            if (f_j <= 0.88437f) {
                                if (f_j <= 0.86343f) {
                                    if (c_j <= 1.04013f) {
                                        votes[1]++
                                    } else {
                                        if (c_j <= 1.04809f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.95325f) {
                                        votes[0]++
                                    } else {
                                        if (h_w <= 1.21001f) {
                                            if (h_w <= 1.19879f) {
                                                if (h_w <= 1.18581f) {
                                                    votes[0]++
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.03895f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.86581f) {
                                                    votes[0]++
                                                } else {
                                                    if (f_j <= 0.86670f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.95120f) {
                                    if (c_j <= 1.05613f) {
                                        if (c_j <= 1.05311f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.91015f) {
                                            votes[3]++
                                        } else {
                                            if (f_j <= 0.92463f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        }
                                    }
                                } else {
                                    if (c_j <= 1.03940f) {
                                        if (f_j <= 0.90569f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (j_w <= 0.95813f) {
                                            if (c_j <= 1.04687f) {
                                                if (c_j <= 1.04440f) {
                                                    votes[4]++
                                                } else {
                                                    if (j_w <= 0.95566f) {
                                                        votes[2]++
                                                    } else {
                                                        if (c_j <= 1.04540f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (c_j <= 1.04787f) {
                                                    votes[1]++
                                                } else {
                                                    if (c_j <= 1.04802f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (c_j <= 1.06846f) {
                                if (j_w <= 0.95237f) {
                                    if (c_j <= 1.05720f) {
                                        if (h_w <= 1.26150f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.26717f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.19170f) {
                    if (c_j <= 1.01951f) {
                        if (c_j <= 1.01801f) {
                            if (f_j <= 0.83049f) {
                                votes[2]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (f_j <= 0.87184f) {
                            if (f_j <= 0.81656f) {
                                votes[0]++
                            } else {
                                if (j_w <= 0.96614f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.16259f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03082f) {
                                            if (c_j <= 1.03047f) {
                                                if (h_w <= 1.18038f) {
                                                    if (c_j <= 1.02857f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.17659f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.96903f) {
                                votes[3]++
                            } else {
                                if (f_j <= 0.88952f) {
                                    if (f_j <= 0.87311f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (h_w <= 1.16052f) {
                                        votes[2]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (f_j <= 0.84981f) {
                        if (f_j <= 0.83033f) {
                            if (c_j <= 1.01149f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.03397f) {
                                    if (h_w <= 1.20053f) {
                                        votes[1]++
                                    } else {
                                        if (j_w <= 0.96986f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97606f) {
                                                votes[4]++
                                            } else {
                                                if (c_j <= 1.01535f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            }
                        } else {
                            if (f_j <= 0.84359f) {
                                if (j_w <= 0.97568f) {
                                    votes[4]++
                                } else {
                                    if (c_j <= 1.02379f) {
                                        if (h_w <= 1.22354f) {
                                            votes[4]++
                                        } else {
                                            if (j_w <= 0.98444f) {
                                                if (h_w <= 1.22670f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.84519f) {
                                    if (c_j <= 1.02143f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.97796f) {
                            if (j_w <= 0.96621f) {
                                if (f_j <= 0.88098f) {
                                    if (c_j <= 1.03549f) {
                                        votes[3]++
                                    } else {
                                        if (c_j <= 1.03584f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (h_w <= 1.23322f) {
                                    if (c_j <= 1.03169f) {
                                        if (c_j <= 1.02449f) {
                                            votes[0]++
                                        } else {
                                            if (j_w <= 0.97409f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.02552f) {
                                                    votes[3]++
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96720f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.96849f) {
                                        if (c_j <= 1.03350f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (h_w <= 1.25736f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.26664f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 44
        if (h_w <= 1.15139f) {
            if (h_w <= 1.12243f) {
                if (f_j <= 0.86430f) {
                    if (j_w <= 0.97971f) {
                        if (h_w <= 1.07034f) {
                            votes[2]++
                        } else {
                            votes[1]++
                        }
                    } else {
                        votes[2]++
                    }
                } else {
                    votes[2]++
                }
            } else {
                if (f_j <= 0.90205f) {
                    if (f_j <= 0.83985f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.14864f) {
                            if (h_w <= 1.14183f) {
                                if (c_j <= 1.05113f) {
                                    if (f_j <= 0.85260f) {
                                        if (f_j <= 0.85068f) {
                                            votes[3]++
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (j_w <= 0.96071f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.14580f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        } else {
                            if (f_j <= 0.85231f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.04589f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.95530f) {
                            votes[2]++
                        } else {
                            votes[0]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.29108f) {
                if (f_j <= 0.88097f) {
                    if (f_j <= 0.84803f) {
                        if (h_w <= 1.18300f) {
                            if (h_w <= 1.17573f) {
                                if (f_j <= 0.84028f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.84118f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                if (h_w <= 1.18198f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (f_j <= 0.83033f) {
                                if (c_j <= 1.01535f) {
                                    if (h_w <= 1.21062f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.80906f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (f_j <= 0.82832f) {
                                        if (j_w <= 0.97770f) {
                                            if (f_j <= 0.82384f) {
                                                if (c_j <= 1.02667f) {
                                                    votes[3]++
                                                } else {
                                                    if (j_w <= 0.96457f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.24732f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                }
                            } else {
                                if (j_w <= 0.98088f) {
                                    if (h_w <= 1.19014f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.20108f) {
                                            if (c_j <= 1.02848f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.03632f) {
                                                if (h_w <= 1.23950f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.25285f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                if (h_w <= 1.20912f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.98488f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.04125f) {
                            if (h_w <= 1.22219f) {
                                if (c_j <= 1.01972f) {
                                    votes[1]++
                                } else {
                                    if (c_j <= 1.03716f) {
                                        if (f_j <= 0.86418f) {
                                            if (j_w <= 0.97032f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.03022f) {
                                                    if (h_w <= 1.20484f) {
                                                        votes[3]++
                                                    } else {
                                                        if (h_w <= 1.21399f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.02686f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.02876f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.16202f) {
                                                        votes[1]++
                                                    } else {
                                                        if (c_j <= 1.03552f) {
                                                            if (f_j <= 0.87296f) {
                                                                if (f_j <= 0.86818f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[4]++
                                                                }
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        } else {
                                                            if (j_w <= 0.96545f) {
                                                                votes[4]++
                                                            } else {
                                                                votes[1]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.96298f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.03212f) {
                                    if (j_w <= 0.97248f) {
                                        if (f_j <= 0.86167f) {
                                            if (c_j <= 1.03139f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    if (f_j <= 0.86383f) {
                                        if (f_j <= 0.85341f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.96704f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.16671f) {
                                if (f_j <= 0.86399f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (j_w <= 0.95400f) {
                                    if (h_w <= 1.19764f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.95806f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.95944f) {
                                            votes[1]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.94883f) {
                        if (j_w <= 0.94533f) {
                            if (j_w <= 0.94319f) {
                                if (h_w <= 1.25512f) {
                                    if (f_j <= 0.90508f) {
                                        votes[1]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (j_w <= 0.93106f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (c_j <= 1.05629f) {
                                votes[4]++
                            } else {
                                if (c_j <= 1.05726f) {
                                    votes[3]++
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    } else {
                        if (h_w <= 1.17911f) {
                            if (f_j <= 0.89867f) {
                                if (f_j <= 0.89023f) {
                                    votes[1]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.90495f) {
                                if (c_j <= 1.05119f) {
                                    if (j_w <= 0.95552f) {
                                        votes[1]++
                                    } else {
                                        if (f_j <= 0.89123f) {
                                            if (f_j <= 0.89012f) {
                                                votes[0]++
                                            } else {
                                                votes[4]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (c_j <= 1.04370f) {
                                    votes[1]++
                                } else {
                                    if (h_w <= 1.23217f) {
                                        if (j_w <= 0.95599f) {
                                            votes[0]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (f_j <= 0.91648f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f_j <= 0.85277f) {
                    votes[4]++
                } else {
                    votes[1]++
                }
            }
        }
        // Tree 45
        if (c_j <= 1.03531f) {
            if (h_w <= 1.14038f) {
                if (f_j <= 0.85711f) {
                    if (c_j <= 1.02567f) {
                        votes[2]++
                    } else {
                        votes[3]++
                    }
                } else {
                    if (c_j <= 1.02572f) {
                        if (h_w <= 1.11713f) {
                            votes[1]++
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (j_w <= 0.96732f) {
                            votes[2]++
                        } else {
                            if (f_j <= 0.87081f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.97286f) {
                    if (h_w <= 1.20266f) {
                        if (j_w <= 0.97020f) {
                            if (f_j <= 0.86842f) {
                                votes[3]++
                            } else {
                                if (c_j <= 1.03414f) {
                                    votes[4]++
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (c_j <= 1.02943f) {
                                if (h_w <= 1.17172f) {
                                    votes[3]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (f_j <= 0.88277f) {
                            if (j_w <= 0.96621f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.22824f) {
                                    if (f_j <= 0.85906f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (f_j <= 0.80906f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.29550f) {
                                            if (f_j <= 0.86322f) {
                                                if (f_j <= 0.84789f) {
                                                    if (j_w <= 0.96694f) {
                                                        votes[1]++
                                                    } else {
                                                        if (h_w <= 1.24797f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (c_j <= 1.03158f) {
                                                    votes[0]++
                                                } else {
                                                    votes[3]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.25778f) {
                                votes[0]++
                            } else {
                                if (c_j <= 1.03283f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.23034f) {
                        if (h_w <= 1.18137f) {
                            if (j_w <= 0.98004f) {
                                if (c_j <= 1.02728f) {
                                    if (j_w <= 0.97526f) {
                                        if (c_j <= 1.02575f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    if (c_j <= 1.02765f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (j_w <= 0.97385f) {
                                votes[3]++
                            } else {
                                if (h_w <= 1.18237f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.86970f) {
                                        if (f_j <= 0.83891f) {
                                            if (h_w <= 1.18831f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.79735f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.81193f) {
                                                        votes[4]++
                                                    } else {
                                                        if (h_w <= 1.21012f) {
                                                            votes[3]++
                                                        } else {
                                                            if (h_w <= 1.22718f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[3]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (f_j <= 0.84981f) {
                                                if (c_j <= 1.02534f) {
                                                    votes[4]++
                                                } else {
                                                    if (j_w <= 0.97444f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.97512f) {
                                                    votes[2]++
                                                } else {
                                                    if (j_w <= 0.97610f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (j_w <= 0.98981f) {
                            if (f_j <= 0.86024f) {
                                if (f_j <= 0.83813f) {
                                    if (f_j <= 0.83405f) {
                                        votes[4]++
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[0]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.15515f) {
                if (h_w <= 1.12243f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.12450f) {
                        votes[0]++
                    } else {
                        if (f_j <= 0.89354f) {
                            if (c_j <= 1.05031f) {
                                if (f_j <= 0.86087f) {
                                    votes[4]++
                                } else {
                                    if (h_w <= 1.15135f) {
                                        if (j_w <= 0.95980f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[4]++
                            }
                        } else {
                            if (c_j <= 1.06053f) {
                                if (c_j <= 1.05402f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.05886f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    }
                }
            } else {
                if (f_j <= 0.85239f) {
                    if (h_w <= 1.23143f) {
                        if (j_w <= 0.96181f) {
                            votes[0]++
                        } else {
                            if (j_w <= 0.96223f) {
                                votes[4]++
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        votes[1]++
                    }
                } else {
                    if (c_j <= 1.03579f) {
                        votes[1]++
                    } else {
                        if (h_w <= 1.28937f) {
                            if (f_j <= 0.90236f) {
                                if (j_w <= 0.95727f) {
                                    if (f_j <= 0.88633f) {
                                        if (h_w <= 1.24933f) {
                                            if (f_j <= 0.88437f) {
                                                if (f_j <= 0.86562f) {
                                                    if (j_w <= 0.95140f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        if (j_w <= 0.94957f) {
                                            votes[4]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.96263f) {
                                        if (j_w <= 0.96067f) {
                                            if (j_w <= 0.95999f) {
                                                if (f_j <= 0.87144f) {
                                                    if (c_j <= 1.04318f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (f_j <= 0.87687f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.21131f) {
                                                votes[3]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    } else {
                                        if (f_j <= 0.87280f) {
                                            if (f_j <= 0.86544f) {
                                                if (j_w <= 0.96383f) {
                                                    votes[1]++
                                                } else {
                                                    if (f_j <= 0.86362f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (h_w <= 1.27019f) {
                                    if (c_j <= 1.03890f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.91159f) {
                                            if (c_j <= 1.03940f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.90351f) {
                                                    votes[3]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.94806f) {
                                                if (h_w <= 1.25768f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (f_j <= 0.91528f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.20614f) {
                                                        votes[1]++
                                                    } else {
                                                        if (h_w <= 1.24205f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 46
        if (c_j <= 1.02069f) {
            if (j_w <= 0.99216f) {
                if (f_j <= 0.81848f) {
                    if (f_j <= 0.81350f) {
                        if (f_j <= 0.79505f) {
                            votes[3]++
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (j_w <= 0.98087f) {
                            votes[3]++
                        } else {
                            votes[0]++
                        }
                    }
                } else {
                    if (f_j <= 0.84352f) {
                        if (f_j <= 0.83903f) {
                            if (h_w <= 1.14753f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.98366f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.83644f) {
                                        votes[3]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            votes[2]++
                        }
                    } else {
                        if (j_w <= 0.98066f) {
                            votes[3]++
                        } else {
                            if (h_w <= 1.09294f) {
                                votes[2]++
                            } else {
                                if (j_w <= 0.98249f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.84646f) {
                                        votes[1]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (j_w <= 0.99367f) {
                    if (c_j <= 1.00715f) {
                        if (j_w <= 0.99334f) {
                            votes[3]++
                        } else {
                            votes[2]++
                        }
                    } else {
                        votes[0]++
                    }
                } else {
                    votes[0]++
                }
            }
        } else {
            if (f_j <= 0.86445f) {
                if (h_w <= 1.22195f) {
                    if (c_j <= 1.03426f) {
                        if (f_j <= 0.82178f) {
                            if (j_w <= 0.97188f) {
                                if (f_j <= 0.81324f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (h_w <= 1.19563f) {
                                if (h_w <= 1.08666f) {
                                    votes[2]++
                                } else {
                                    if (j_w <= 0.97103f) {
                                        if (c_j <= 1.03082f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (c_j <= 1.02453f) {
                                            if (f_j <= 0.84429f) {
                                                votes[3]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.02705f) {
                                    if (f_j <= 0.84779f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.02530f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                } else {
                                    if (h_w <= 1.20032f) {
                                        if (h_w <= 1.19837f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            }
                        }
                    } else {
                        if (f_j <= 0.83957f) {
                            votes[2]++
                        } else {
                            if (c_j <= 1.03532f) {
                                votes[0]++
                            } else {
                                if (h_w <= 1.16727f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.96458f) {
                                        if (c_j <= 1.04564f) {
                                            if (c_j <= 1.03931f) {
                                                if (f_j <= 0.84919f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (h_w <= 1.17939f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (j_w <= 0.96750f) {
                        votes[4]++
                    } else {
                        if (c_j <= 1.03316f) {
                            if (h_w <= 1.23034f) {
                                if (f_j <= 0.84766f) {
                                    votes[4]++
                                } else {
                                    if (f_j <= 0.85344f) {
                                        votes[0]++
                                    } else {
                                        votes[2]++
                                    }
                                }
                            } else {
                                if (f_j <= 0.85090f) {
                                    if (j_w <= 0.97568f) {
                                        votes[4]++
                                    } else {
                                        if (f_j <= 0.84261f) {
                                            votes[1]++
                                        } else {
                                            votes[4]++
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.97121f) {
                                        votes[3]++
                                    } else {
                                        votes[1]++
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.23493f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            } else {
                if (h_w <= 1.15492f) {
                    if (h_w <= 1.12198f) {
                        votes[2]++
                    } else {
                        if (h_w <= 1.12433f) {
                            votes[0]++
                        } else {
                            if (f_j <= 0.90205f) {
                                if (j_w <= 0.95210f) {
                                    votes[4]++
                                } else {
                                    if (j_w <= 0.95913f) {
                                        if (f_j <= 0.89679f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.03062f) {
                        if (f_j <= 0.87120f) {
                            if (h_w <= 1.15719f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        } else {
                            votes[1]++
                        }
                    } else {
                        if (h_w <= 1.29108f) {
                            if (f_j <= 0.89950f) {
                                if (j_w <= 0.96730f) {
                                    if (h_w <= 1.26216f) {
                                        if (c_j <= 1.04905f) {
                                            if (j_w <= 0.95460f) {
                                                votes[3]++
                                            } else {
                                                if (h_w <= 1.22845f) {
                                                    if (f_j <= 0.86537f) {
                                                        votes[0]++
                                                    } else {
                                                        if (f_j <= 0.86723f) {
                                                            votes[1]++
                                                        } else {
                                                            if (f_j <= 0.88766f) {
                                                                if (c_j <= 1.04212f) {
                                                                    if (h_w <= 1.17979f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        votes[3]++
                                                                    }
                                                                } else {
                                                                    if (f_j <= 0.87144f) {
                                                                        votes[0]++
                                                                    } else {
                                                                        votes[4]++
                                                                    }
                                                                }
                                                            } else {
                                                                if (f_j <= 0.89023f) {
                                                                    votes[1]++
                                                                } else {
                                                                    votes[2]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.03819f) {
                                                        votes[0]++
                                                    } else {
                                                        if (f_j <= 0.89012f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.24397f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.94567f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    if (f_j <= 0.87417f) {
                                        if (f_j <= 0.86792f) {
                                            votes[3]++
                                        } else {
                                            votes[4]++
                                        }
                                    } else {
                                        if (j_w <= 0.96887f) {
                                            votes[3]++
                                        } else {
                                            votes[0]++
                                        }
                                    }
                                }
                            } else {
                                if (c_j <= 1.05128f) {
                                    if (h_w <= 1.17391f) {
                                        votes[1]++
                                    } else {
                                        if (h_w <= 1.18606f) {
                                            if (j_w <= 0.96002f) {
                                                votes[4]++
                                            } else {
                                                votes[3]++
                                            }
                                        } else {
                                            if (c_j <= 1.04317f) {
                                                votes[0]++
                                            } else {
                                                if (f_j <= 0.91808f) {
                                                    votes[1]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.94685f) {
                                        if (h_w <= 1.21248f) {
                                            if (f_j <= 0.92786f) {
                                                votes[0]++
                                            } else {
                                                votes[2]++
                                            }
                                        } else {
                                            if (h_w <= 1.26768f) {
                                                if (j_w <= 0.94429f) {
                                                    votes[1]++
                                                } else {
                                                    votes[3]++
                                                }
                                            } else {
                                                if (f_j <= 0.95069f) {
                                                    votes[4]++
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                }
                            }
                        } else {
                            votes[1]++
                        }
                    }
                }
            }
        }
        // Tree 47
        if (h_w <= 1.14747f) {
            if (j_w <= 0.95978f) {
                if (h_w <= 1.11663f) {
                    votes[2]++
                } else {
                    if (h_w <= 1.12433f) {
                        votes[0]++
                    } else {
                        votes[2]++
                    }
                }
            } else {
                if (c_j <= 1.02209f) {
                    votes[2]++
                } else {
                    if (j_w <= 0.96232f) {
                        if (j_w <= 0.96117f) {
                            votes[3]++
                        } else {
                            votes[4]++
                        }
                    } else {
                        if (c_j <= 1.03695f) {
                            if (c_j <= 1.03554f) {
                                if (f_j <= 0.87020f) {
                                    if (j_w <= 0.97238f) {
                                        votes[3]++
                                    } else {
                                        if (f_j <= 0.85722f) {
                                            votes[2]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[2]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            votes[2]++
                        }
                    }
                }
            }
        } else {
            if (h_w <= 1.30816f) {
                if (j_w <= 0.96719f) {
                    if (c_j <= 1.03727f) {
                        if (j_w <= 0.96444f) {
                            votes[1]++
                        } else {
                            if (h_w <= 1.25125f) {
                                if (j_w <= 0.96540f) {
                                    votes[3]++
                                } else {
                                    if (c_j <= 1.03576f) {
                                        if (f_j <= 0.86966f) {
                                            if (h_w <= 1.22098f) {
                                                votes[0]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (h_w <= 1.20029f) {
                                            votes[4]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                votes[0]++
                            }
                        }
                    } else {
                        if (h_w <= 1.15923f) {
                            if (h_w <= 1.15201f) {
                                if (f_j <= 0.86664f) {
                                    votes[2]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (c_j <= 1.05495f) {
                                    if (j_w <= 0.96299f) {
                                        if (h_w <= 1.15860f) {
                                            if (h_w <= 1.15521f) {
                                                votes[2]++
                                            } else {
                                                votes[1]++
                                            }
                                        } else {
                                            votes[2]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            }
                        } else {
                            if (h_w <= 1.17376f) {
                                if (j_w <= 0.95965f) {
                                    votes[4]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (j_w <= 0.96383f) {
                                    if (h_w <= 1.22018f) {
                                        if (h_w <= 1.21587f) {
                                            if (j_w <= 0.94574f) {
                                                votes[2]++
                                            } else {
                                                if (j_w <= 0.95341f) {
                                                    if (j_w <= 0.94639f) {
                                                        votes[3]++
                                                    } else {
                                                        if (f_j <= 0.86917f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                } else {
                                                    if (j_w <= 0.95573f) {
                                                        votes[1]++
                                                    } else {
                                                        if (f_j <= 0.87274f) {
                                                            if (c_j <= 1.03976f) {
                                                                votes[1]++
                                                            } else {
                                                                if (h_w <= 1.17560f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (h_w <= 1.20107f) {
                                                                        if (f_j <= 0.84153f) {
                                                                            votes[0]++
                                                                        } else {
                                                                            votes[3]++
                                                                        }
                                                                    } else {
                                                                        votes[1]++
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            if (c_j <= 1.04169f) {
                                                                if (f_j <= 0.89640f) {
                                                                    votes[1]++
                                                                } else {
                                                                    votes[0]++
                                                                }
                                                            } else {
                                                                if (f_j <= 0.92123f) {
                                                                    votes[4]++
                                                                } else {
                                                                    votes[3]++
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            votes[0]++
                                        }
                                    } else {
                                        if (c_j <= 1.04092f) {
                                            votes[4]++
                                        } else {
                                            if (h_w <= 1.26243f) {
                                                if (h_w <= 1.25629f) {
                                                    if (c_j <= 1.04276f) {
                                                        votes[1]++
                                                    } else {
                                                        if (j_w <= 0.95220f) {
                                                            if (h_w <= 1.25094f) {
                                                                if (j_w <= 0.94084f) {
                                                                    votes[0]++
                                                                } else {
                                                                    if (f_j <= 0.89550f) {
                                                                        votes[1]++
                                                                    } else {
                                                                        if (h_w <= 1.23957f) {
                                                                            votes[0]++
                                                                        } else {
                                                                            votes[1]++
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            if (c_j <= 1.04779f) {
                                                                votes[0]++
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (h_w <= 1.28582f) {
                                                    if (j_w <= 0.94867f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.19640f) {
                        if (f_j <= 0.87184f) {
                            if (f_j <= 0.82093f) {
                                if (c_j <= 1.02687f) {
                                    votes[0]++
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (f_j <= 0.85695f) {
                                    votes[3]++
                                } else {
                                    if (h_w <= 1.16320f) {
                                        if (f_j <= 0.85918f) {
                                            votes[0]++
                                        } else {
                                            if (c_j <= 1.02963f) {
                                                votes[3]++
                                            } else {
                                                votes[1]++
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.97687f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        } else {
                            if (h_w <= 1.16629f) {
                                votes[1]++
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (c_j <= 1.01069f) {
                            if (j_w <= 0.99341f) {
                                votes[3]++
                            } else {
                                votes[0]++
                            }
                        } else {
                            if (j_w <= 0.96840f) {
                                if (j_w <= 0.96780f) {
                                    if (f_j <= 0.86343f) {
                                        votes[4]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (f_j <= 0.84934f) {
                                    if (j_w <= 0.96988f) {
                                        if (j_w <= 0.96897f) {
                                            votes[0]++
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (f_j <= 0.83852f) {
                                            if (f_j <= 0.82208f) {
                                                if (c_j <= 1.02455f) {
                                                    if (f_j <= 0.81193f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[0]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (h_w <= 1.22670f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.23005f) {
                                                        votes[3]++
                                                    } else {
                                                        if (f_j <= 0.83405f) {
                                                            votes[4]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (j_w <= 0.98385f) {
                                                votes[4]++
                                            } else {
                                                if (f_j <= 0.84112f) {
                                                    votes[4]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (j_w <= 0.96955f) {
                                        votes[0]++
                                    } else {
                                        if (f_j <= 0.88337f) {
                                            if (j_w <= 0.97134f) {
                                                votes[3]++
                                            } else {
                                                if (c_j <= 1.02802f) {
                                                    if (j_w <= 0.97460f) {
                                                        votes[3]++
                                                    } else {
                                                        if (f_j <= 0.85255f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[3]++
                                                        }
                                                    }
                                                } else {
                                                    votes[0]++
                                                }
                                            }
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                votes[1]++
            }
        }
        // Tree 48
        if (h_w <= 1.15921f) {
            if (f_j <= 0.89064f) {
                if (j_w <= 0.97987f) {
                    if (c_j <= 1.03473f) {
                        if (h_w <= 1.12358f) {
                            if (f_j <= 0.87009f) {
                                votes[1]++
                            } else {
                                votes[2]++
                            }
                        } else {
                            if (f_j <= 0.87184f) {
                                votes[3]++
                            } else {
                                votes[1]++
                            }
                        }
                    } else {
                        if (j_w <= 0.95346f) {
                            if (h_w <= 1.12912f) {
                                if (c_j <= 1.05373f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (f_j <= 0.88702f) {
                                if (j_w <= 0.96441f) {
                                    votes[2]++
                                } else {
                                    if (c_j <= 1.03578f) {
                                        votes[2]++
                                    } else {
                                        votes[3]++
                                    }
                                }
                            } else {
                                votes[3]++
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.14050f) {
                        votes[2]++
                    } else {
                        if (j_w <= 0.98394f) {
                            votes[2]++
                        } else {
                            votes[3]++
                        }
                    }
                }
            } else {
                if (h_w <= 1.15628f) {
                    votes[2]++
                } else {
                    if (c_j <= 1.04315f) {
                        votes[1]++
                    } else {
                        votes[2]++
                    }
                }
            }
        } else {
            if (h_w <= 1.30735f) {
                if (h_w <= 1.20783f) {
                    if (c_j <= 1.03574f) {
                        if (h_w <= 1.19740f) {
                            if (f_j <= 0.85441f) {
                                if (j_w <= 0.97452f) {
                                    if (c_j <= 1.02697f) {
                                        votes[0]++
                                    } else {
                                        votes[3]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            } else {
                                if (j_w <= 0.97000f) {
                                    votes[3]++
                                } else {
                                    if (f_j <= 0.85961f) {
                                        votes[0]++
                                    } else {
                                        if (j_w <= 0.97297f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.17129f) {
                                                votes[1]++
                                            } else {
                                                votes[4]++
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (j_w <= 0.97606f) {
                                if (j_w <= 0.97455f) {
                                    if (h_w <= 1.19919f) {
                                        votes[4]++
                                    } else {
                                        if (c_j <= 1.03202f) {
                                            votes[3]++
                                        } else {
                                            if (h_w <= 1.20567f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[4]++
                            }
                        }
                    } else {
                        if (f_j <= 0.89000f) {
                            if (h_w <= 1.19087f) {
                                if (j_w <= 0.96004f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.86489f) {
                                        if (f_j <= 0.83309f) {
                                            votes[0]++
                                        } else {
                                            if (f_j <= 0.85589f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        }
                                    } else {
                                        if (c_j <= 1.04115f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                }
                            } else {
                                if (j_w <= 0.95905f) {
                                    if (j_w <= 0.95636f) {
                                        if (c_j <= 1.04822f) {
                                            votes[0]++
                                        } else {
                                            if (h_w <= 1.19764f) {
                                                votes[0]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[1]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (h_w <= 1.18606f) {
                                if (f_j <= 0.92079f) {
                                    if (c_j <= 1.04481f) {
                                        votes[4]++
                                    } else {
                                        if (h_w <= 1.17653f) {
                                            votes[4]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                } else {
                                    votes[1]++
                                }
                            } else {
                                if (h_w <= 1.19026f) {
                                    votes[0]++
                                } else {
                                    if (h_w <= 1.19892f) {
                                        votes[2]++
                                    } else {
                                        if (j_w <= 0.95576f) {
                                            votes[1]++
                                        } else {
                                            votes[3]++
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (h_w <= 1.22181f) {
                        if (c_j <= 1.05665f) {
                            if (h_w <= 1.20901f) {
                                votes[1]++
                            } else {
                                if (c_j <= 1.04205f) {
                                    votes[0]++
                                } else {
                                    if (f_j <= 0.91482f) {
                                        votes[1]++
                                    } else {
                                        votes[0]++
                                    }
                                }
                            }
                        } else {
                            votes[3]++
                        }
                    } else {
                        if (c_j <= 1.06846f) {
                            if (h_w <= 1.25877f) {
                                if (f_j <= 0.91047f) {
                                    if (f_j <= 0.84359f) {
                                        if (j_w <= 0.96790f) {
                                            if (h_w <= 1.23493f) {
                                                votes[1]++
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            if (j_w <= 0.97857f) {
                                                if (c_j <= 1.02492f) {
                                                    if (f_j <= 0.84059f) {
                                                        votes[1]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (f_j <= 0.82766f) {
                                                    votes[0]++
                                                } else {
                                                    if (c_j <= 1.01581f) {
                                                        votes[4]++
                                                    } else {
                                                        if (j_w <= 0.98126f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (h_w <= 1.22401f) {
                                            votes[4]++
                                        } else {
                                            if (c_j <= 1.02900f) {
                                                votes[1]++
                                            } else {
                                                if (j_w <= 0.96770f) {
                                                    if (c_j <= 1.04125f) {
                                                        if (f_j <= 0.87882f) {
                                                            votes[1]++
                                                        } else {
                                                            votes[4]++
                                                        }
                                                    } else {
                                                        if (c_j <= 1.05397f) {
                                                            if (h_w <= 1.23106f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        } else {
                                                            if (c_j <= 1.06245f) {
                                                                votes[1]++
                                                            } else {
                                                                votes[0]++
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    if (c_j <= 1.03022f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.24214f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            } else {
                                if (c_j <= 1.03713f) {
                                    if (f_j <= 0.86146f) {
                                        votes[4]++
                                    } else {
                                        if (j_w <= 0.97021f) {
                                            votes[0]++
                                        } else {
                                            votes[1]++
                                        }
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (f_j <= 0.95115f) {
                                votes[1]++
                            } else {
                                votes[0]++
                            }
                        }
                    }
                }
            } else {
                votes[1]++
            }
        }
        // Tree 49
        if (h_w <= 1.12198f) {
            if (c_j <= 1.02165f) {
                votes[2]++
            } else {
                if (c_j <= 1.02572f) {
                    votes[1]++
                } else {
                    votes[2]++
                }
            }
        } else {
            if (j_w <= 0.96399f) {
                if (h_w <= 1.29108f) {
                    if (h_w <= 1.15515f) {
                        if (c_j <= 1.04188f) {
                            if (c_j <= 1.03978f) {
                                votes[4]++
                            } else {
                                votes[3]++
                            }
                        } else {
                            if (f_j <= 0.91498f) {
                                if (f_j <= 0.91042f) {
                                    if (c_j <= 1.05109f) {
                                        if (f_j <= 0.86842f) {
                                            votes[2]++
                                        } else {
                                            if (h_w <= 1.14486f) {
                                                votes[2]++
                                            } else {
                                                votes[3]++
                                            }
                                        }
                                    } else {
                                        votes[4]++
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                if (f_j <= 0.92493f) {
                                    if (j_w <= 0.94977f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[2]++
                                }
                            }
                        }
                    } else {
                        if (c_j <= 1.03958f) {
                            if (f_j <= 0.86785f) {
                                if (h_w <= 1.19841f) {
                                    votes[1]++
                                } else {
                                    if (f_j <= 0.84267f) {
                                        votes[0]++
                                    } else {
                                        votes[4]++
                                    }
                                }
                            } else {
                                if (c_j <= 1.03920f) {
                                    if (f_j <= 0.88045f) {
                                        if (j_w <= 0.96298f) {
                                            votes[3]++
                                        } else {
                                            votes[1]++
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    votes[4]++
                                }
                            }
                        } else {
                            if (h_w <= 1.27077f) {
                                if (f_j <= 0.88920f) {
                                    if (f_j <= 0.88064f) {
                                        if (j_w <= 0.95967f) {
                                            if (h_w <= 1.20109f) {
                                                if (c_j <= 1.04509f) {
                                                    votes[4]++
                                                } else {
                                                    if (h_w <= 1.19764f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                }
                                            } else {
                                                if (f_j <= 0.86432f) {
                                                    votes[1]++
                                                } else {
                                                    if (h_w <= 1.24122f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (c_j <= 1.04100f) {
                                                votes[0]++
                                            } else {
                                                if (j_w <= 0.96038f) {
                                                    votes[0]++
                                                } else {
                                                    votes[1]++
                                                }
                                            }
                                        }
                                    } else {
                                        votes[0]++
                                    }
                                } else {
                                    if (j_w <= 0.95426f) {
                                        if (j_w <= 0.94880f) {
                                            if (h_w <= 1.22414f) {
                                                if (j_w <= 0.94685f) {
                                                    if (f_j <= 0.92786f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[2]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            } else {
                                                if (f_j <= 0.89382f) {
                                                    votes[1]++
                                                } else {
                                                    if (f_j <= 0.90319f) {
                                                        votes[4]++
                                                    } else {
                                                        votes[1]++
                                                    }
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.19602f) {
                                                votes[1]++
                                            } else {
                                                if (f_j <= 0.90717f) {
                                                    votes[0]++
                                                } else {
                                                    if (h_w <= 1.23217f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (j_w <= 0.95514f) {
                                            votes[1]++
                                        } else {
                                            if (h_w <= 1.16945f) {
                                                if (f_j <= 0.89867f) {
                                                    votes[2]++
                                                } else {
                                                    votes[1]++
                                                }
                                            } else {
                                                if (f_j <= 0.90871f) {
                                                    if (j_w <= 0.95711f) {
                                                        votes[3]++
                                                    } else {
                                                        votes[4]++
                                                    }
                                                } else {
                                                    votes[4]++
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (f_j <= 0.95069f) {
                                    votes[4]++
                                } else {
                                    votes[0]++
                                }
                            }
                        }
                    }
                } else {
                    votes[1]++
                }
            } else {
                if (h_w <= 1.17858f) {
                    if (f_j <= 0.86460f) {
                        if (f_j <= 0.85709f) {
                            if (c_j <= 1.01990f) {
                                if (c_j <= 1.01069f) {
                                    votes[3]++
                                } else {
                                    votes[2]++
                                }
                            } else {
                                if (h_w <= 1.13403f) {
                                    if (f_j <= 0.85068f) {
                                        votes[3]++
                                    } else {
                                        votes[2]++
                                    }
                                } else {
                                    votes[3]++
                                }
                            }
                        } else {
                            if (f_j <= 0.86031f) {
                                votes[0]++
                            } else {
                                votes[3]++
                            }
                        }
                    } else {
                        if (c_j <= 1.03072f) {
                            if (c_j <= 1.02900f) {
                                if (h_w <= 1.16177f) {
                                    votes[3]++
                                } else {
                                    votes[1]++
                                }
                            } else {
                                votes[1]++
                            }
                        } else {
                            if (f_j <= 0.89214f) {
                                votes[3]++
                            } else {
                                votes[2]++
                            }
                        }
                    }
                } else {
                    if (c_j <= 1.01149f) {
                        votes[0]++
                    } else {
                        if (c_j <= 1.01512f) {
                            votes[4]++
                        } else {
                            if (h_w <= 1.31410f) {
                                if (f_j <= 0.88658f) {
                                    if (h_w <= 1.19198f) {
                                        if (j_w <= 0.98122f) {
                                            if (c_j <= 1.03344f) {
                                                if (j_w <= 0.97094f) {
                                                    votes[3]++
                                                } else {
                                                    if (f_j <= 0.84057f) {
                                                        votes[0]++
                                                    } else {
                                                        if (h_w <= 1.18858f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[0]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                votes[0]++
                                            }
                                        } else {
                                            votes[3]++
                                        }
                                    } else {
                                        if (f_j <= 0.84934f) {
                                            if (h_w <= 1.19563f) {
                                                votes[3]++
                                            } else {
                                                if (f_j <= 0.81753f) {
                                                    votes[0]++
                                                } else {
                                                    if (j_w <= 0.96592f) {
                                                        votes[1]++
                                                    } else {
                                                        if (f_j <= 0.83891f) {
                                                            if (c_j <= 1.02608f) {
                                                                if (c_j <= 1.02267f) {
                                                                    votes[3]++
                                                                } else {
                                                                    votes[1]++
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        } else {
                                                            if (f_j <= 0.84094f) {
                                                                if (c_j <= 1.02757f) {
                                                                    votes[4]++
                                                                } else {
                                                                    votes[3]++
                                                                }
                                                            } else {
                                                                votes[4]++
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            if (h_w <= 1.22390f) {
                                                if (f_j <= 0.86387f) {
                                                    if (f_j <= 0.85006f) {
                                                        votes[0]++
                                                    } else {
                                                        votes[3]++
                                                    }
                                                } else {
                                                    if (f_j <= 0.87137f) {
                                                        votes[4]++
                                                    } else {
                                                        if (j_w <= 0.97455f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                }
                                            } else {
                                                if (j_w <= 0.97316f) {
                                                    if (c_j <= 1.03212f) {
                                                        if (j_w <= 0.97182f) {
                                                            votes[0]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    } else {
                                                        if (c_j <= 1.03437f) {
                                                            votes[3]++
                                                        } else {
                                                            votes[1]++
                                                        }
                                                    }
                                                } else {
                                                    votes[2]++
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    votes[0]++
                                }
                            } else {
                                votes[1]++
                            }
                        }
                    }
                }
            }
        }
    }
}