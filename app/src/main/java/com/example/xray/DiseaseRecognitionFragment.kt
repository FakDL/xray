package com.example.xray

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.fragment.app.Fragment
import com.example.xray.databinding.FragmentDiseaseRecognitionBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.Manifest
import android.os.Environment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup


class DiseaseRecognitionFragment : Fragment() {
    private lateinit var binding: FragmentDiseaseRecognitionBinding
    private lateinit var photoFile: File
    private lateinit var imageUri: Uri

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        photoFile = getPhotoFile()
        imageUri = FileProvider.getUriForFile(requireActivity(), "${requireActivity().packageName}.fileprovider", photoFile)

        binding = FragmentDiseaseRecognitionBinding.inflate(inflater, container, false)
        setSpinnerList()
        binding.btnCamera.setOnClickListener {
            askForCameraPermission()
        }
        binding.btnGallery.setOnClickListener {
            askForGalleryPermission()
        }
        return binding.root
    }

    private val cameraPermissionResultLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            takePictureLauncher.launch(imageUri)
        }
    }

    private val galleryPermissionResultLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) selectImageLauncher.launch("image/*")
    }

    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { isSuccess ->
        if (isSuccess) {
            val image = preprocessTakenImage(getImageFromUri(imageUri))
            makePrediction(image)
        }
    }

    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { selectedImageUri ->
        if (selectedImageUri != null) {
            try {
                makePrediction(getImageFromUri(selectedImageUri))
            } catch (_: Exception) {
            }
        }
    }

    private fun askForCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                requireActivity(),
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            cameraPermissionResultLauncher.launch(Manifest.permission.CAMERA)
        } else {
            takePictureLauncher.launch(imageUri)
        }
    }


    private fun askForGalleryPermission() {
        if (ContextCompat.checkSelfPermission(
                requireActivity(),
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            galleryPermissionResultLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        } else {
            selectImageLauncher.launch("image/*")
        }
    }

    private fun setSpinnerList() {
        val items = requireActivity().assets.list("models/")?.toList()
        val arrayAdapter =
            ArrayAdapter(requireActivity(), R.layout.item_spinner, items.orEmpty())
        binding.spinner.adapter = arrayAdapter
    }

    private fun getPhotoFile(): File {
        val photoFileName = "photo.jpg"
        val storageDirectory = requireActivity().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return if (storageDirectory != null) {
            File.createTempFile(photoFileName, ".jpg", storageDirectory)
        } else {
            throw IllegalStateException("External storage not available")
        }
    }

    private fun getImageFromUri(selectedImageUri: Uri): Bitmap {
        val inputStream = requireActivity().contentResolver.openInputStream(selectedImageUri)
        return BitmapFactory.decodeStream(inputStream)
    }

    private fun preprocessTakenImage(image: Bitmap): Bitmap {
        val exif = ExifInterface(photoFile.absolutePath)
        val orientation =
            exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        }
        return Bitmap.createBitmap(image, 0, 0, image.width, image.height, matrix, true)
    }

    private fun compressBitmap(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val newWidth = 300
        val newHeight = 300

        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height

        val matrix = Matrix()

        matrix.postScale(scaleWidth, scaleHeight)

        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false)
    }

    private fun makePrediction(image: Bitmap) {
//        val cropSize = Integer.min(image.width, image.height)
//        val croppedImage = Bitmap.createBitmap(
//            image,
//            (image.width - cropSize) / 2,
//            (image.height - cropSize) / 2,
//            cropSize,
//            cropSize
//        )
        binding.tvPrediction.text = EMPTY_STRING

        val compressedImage = compressBitmap(image)

        binding.imageView.setImageBitmap(image)

        val model = Interpreter(loadModelFile())
        val inputImageBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 300, 300, 3), DataType.FLOAT32)
        val arr = convertBitmapToFloatArray(compressedImage)
        val outputTensor = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
        preprocessImage(arr).also { inputImageBuffer.loadArray(it) }
        model.run(inputImageBuffer.buffer, outputTensor.buffer.rewind())

        val probabilities = outputTensor.floatArray

        val index = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        val classes = arrayOf("COVID-19", "болезнь не найдена", "пневмония", "пневмония")
        val predictedLabel = classes[index]

        binding.tvPrediction.text = getString(R.string.prediction, predictedLabel)

        model.close()
    }

    private fun preprocessImage(arr: FloatArray): FloatArray {
        val mean = floatArrayOf(0f, 0f, 0f)

        for (i in arr.indices step 3) {
            arr[i] -= mean[0]
            arr[i + 1] -= mean[1]
            arr[i + 2] -= mean[2]
            arr[i] /= 255f
            arr[i + 1] /= 255f
            arr[i + 2] /= 255f
        }
        return arr
    }

    private fun convertBitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val floatValues = FloatArray(IMAGE_SIZE * IMAGE_SIZE * CHANNELS)

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val pixel = bitmap.getPixel(x, y)

                val r = Color.red(pixel).toFloat()
                val g = Color.green(pixel).toFloat()
                val b = Color.blue(pixel).toFloat()

                floatValues[(y * IMAGE_SIZE + x) * CHANNELS] = r
                floatValues[(y * IMAGE_SIZE + x) * CHANNELS + 1] = g
                floatValues[(y * IMAGE_SIZE + x) * CHANNELS + 2] = b
            }
        }

        return floatValues
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val filename = "models/" + binding.spinner.selectedItem.toString()
        val fileDescriptor = requireActivity().assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    companion object {
        const val IMAGE_SIZE = 300
        const val CHANNELS = 3
        const val EMPTY_STRING = ""

        fun getInstance() = DiseaseRecognitionFragment()
    }

}
