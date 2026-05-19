package com.kd.classifier

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.kd.classifier.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: Classifier

    companion object {
        private const val PERM_CAMERA = 100
        private const val REQ_GALLERY = 200
        private const val REQ_CAMERA  = 300
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding    = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        classifier = Classifier(this)

        binding.btnCamera.setOnClickListener { takePhoto() }
        binding.btnUpload.setOnClickListener { pickFromGallery() }
    }

    // ── Take photo ─────────────────────────────────────────────────────────────
    private fun takePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), PERM_CAMERA)
            return
        }
        startActivityForResult(Intent(MediaStore.ACTION_IMAGE_CAPTURE), REQ_CAMERA)
    }

    // ── Pick from gallery ──────────────────────────────────────────────────────
    private fun pickFromGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, REQ_GALLERY)
    }

    // ── Handle result ──────────────────────────────────────────────────────────
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != Activity.RESULT_OK) return

        val bitmap: Bitmap = when (requestCode) {
            REQ_CAMERA  -> (data?.extras?.get("data") as? Bitmap) ?: return
            REQ_GALLERY -> {
                val uri = data?.data ?: return
                uriToBitmap(uri) ?: return
            }
            else -> return
        }

        // Show image, hide placeholder, show loading
        binding.ivPhoto.setImageBitmap(bitmap)
        binding.ivPhoto.visibility       = View.VISIBLE
        binding.tvPlaceholder.visibility = View.GONE
        binding.tvClassifying.visibility = View.VISIBLE
        binding.resultsCard.visibility   = View.INVISIBLE

        // Classify on background thread
        Thread {
            val start   = System.currentTimeMillis()
            val results = classifier.classify(bitmap)
            val ms      = System.currentTimeMillis() - start
            runOnUiThread { updateUI(results, ms) }
        }.start()
    }

    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
        } catch (e: Exception) { null }
    }

    // ── Update UI ──────────────────────────────────────────────────────────────
    private fun updateUI(results: List<Result>, ms: Long) {
        binding.tvClassifying.visibility = View.GONE
        binding.resultsCard.visibility   = View.VISIBLE

        if (results.size < 3) return

        val pct  = { f: Float -> "${(f * 100).toInt()}%" }
        val prog = { f: Float -> (f * 100).toInt() }

        binding.tvTop1Label.text = "🥇  ${results[0].label}"
        binding.tvTop1Pct.text   = pct(results[0].confidence)
        binding.pb1.progress     = prog(results[0].confidence)

        binding.tvTop2Label.text = "🥈  ${results[1].label}"
        binding.tvTop2Pct.text   = pct(results[1].confidence)
        binding.pb2.progress     = prog(results[1].confidence)

        binding.tvTop3Label.text = "🥉  ${results[2].label}"
        binding.tvTop3Pct.text   = pct(results[2].confidence)
        binding.pb3.progress     = prog(results[2].confidence)

        binding.tvMs.text = "⚡ ${ms}ms"
    }

    // ── Permissions ────────────────────────────────────────────────────────────
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == PERM_CAMERA &&
            results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            takePhoto()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}