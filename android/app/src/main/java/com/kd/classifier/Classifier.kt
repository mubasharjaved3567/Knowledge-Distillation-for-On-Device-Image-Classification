package com.kd.classifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.min

data class Result(val label: String, val confidence: Float)

class Classifier(ctx: Context) {

    private val module = LiteModuleLoader.load(assetFilePath(ctx, "student_kd.ptl"))

    // CIFAR-100 normalization constants
    private val MEAN = floatArrayOf(0.5071f, 0.4865f, 0.4409f)
    private val STD  = floatArrayOf(0.2675f, 0.2640f, 0.2633f)
    private val IMG_SIZE = 32

    val LABELS = listOf(
        "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee",
        "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
        "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
        "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
        "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
        "keyboard", "lamp", "lawn mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree",
        "pear", "pickup truck", "pine tree", "plain", "plate", "poppy",
        "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
        "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
        "sweet pepper", "table", "tank", "telephone", "television", "tiger",
        "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
        "willow tree", "wolf", "woman", "worm"
    )

    fun classify(bitmap: Bitmap, topK: Int = 3): List<Result> {
        // Step 1: center crop to square
        val size   = min(bitmap.width, bitmap.height)
        val startX = (bitmap.width  - size) / 2
        val startY = (bitmap.height - size) / 2
        val cropped = Bitmap.createBitmap(bitmap, startX, startY, size, size)

        // Step 2: resize to 32x32
        val resized = Bitmap.createScaledBitmap(cropped, IMG_SIZE, IMG_SIZE, true)

        // Step 3: convert to float tensor [1, 3, 32, 32] manually
        // This avoids any channel order issues with TensorImageUtils
        val floatArray = FloatArray(3 * IMG_SIZE * IMG_SIZE)
        for (y in 0 until IMG_SIZE) {
            for (x in 0 until IMG_SIZE) {
                val pixel = resized.getPixel(x, y)
                val r = Color.red(pixel)   / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel)  / 255f
                // CHW format: channel first
                floatArray[0 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (r - MEAN[0]) / STD[0]
                floatArray[1 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (g - MEAN[1]) / STD[1]
                floatArray[2 * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = (b - MEAN[2]) / STD[2]
            }
        }

        // Step 4: create tensor and run inference
        val shape  = longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        val tensor = Tensor.fromBlob(floatArray, shape)
        val output = module.forward(IValue.from(tensor)).toTensor()
        val logits = output.dataAsFloatArray

        // Step 5: softmax
        val maxL = logits.max()!!
        val exps = logits.map { exp((it - maxL).toDouble()) }
        val sumE = exps.sum()
        val probs = exps.map { (it / sumE).toFloat() }

        return probs.indices
            .sortedByDescending { probs[it] }
            .take(topK)
            .map { Result(LABELS[it], probs[it]) }
    }

    private fun assetFilePath(ctx: Context, name: String): String {
        val f = File(ctx.filesDir, name)
        if (!f.exists()) {
            ctx.assets.open(name).use { ins ->
                FileOutputStream(f).use { outs -> ins.copyTo(outs) }
            }
        }
        return f.absolutePath
    }

    fun close() = module.destroy()
}