package com.example.xray.view.ui

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.fragment.app.Fragment
import com.example.xray.databinding.FragmentDiseaseRecognitionBinding
import java.io.File
import android.Manifest
import android.os.Environment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.example.xray.R
import com.example.xray.view.viewmodel.DiseaseRecognitionViewModel
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.launch


class DiseaseRecognitionFragment : Fragment() {
    private lateinit var binding: FragmentDiseaseRecognitionBinding
    private lateinit var photoFile: File
    private lateinit var imageUri: Uri

    private val viewModel: DiseaseRecognitionViewModel by lazy {
        ViewModelProvider(this)[DiseaseRecognitionViewModel::class.java]
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

    private val takePictureLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { isSuccess ->
            if (isSuccess) {
                val image = preprocessTakenImage(getImageFromUri(imageUri))
                makePrediction(image)
            }
        }

    private val selectImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { selectedImageUri ->
            if (selectedImageUri != null) {
                val image = preprocessTakenImage(getImageFromUri(selectedImageUri))
                makePrediction(image)
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        photoFile = getPhotoFile()
        imageUri = FileProvider.getUriForFile(
            requireActivity(),
            "${requireActivity().packageName}.fileprovider",
            photoFile
        )

        binding = FragmentDiseaseRecognitionBinding.inflate(inflater, container, false)
        setSpinnerList()
        binding.btnCamera.setOnClickListener {
            askForCameraPermission()
        }
        binding.btnGallery.setOnClickListener {
            askForGalleryPermission()
        }

        observePrediction()
        return binding.root
    }


    private fun observePrediction() {
        lifecycleScope.launch {
            viewModel.predictedLabel.filter { it.isNotEmpty() }.collect {
                binding.tvPrediction.text = getString(R.string.prediction, it)
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

    private fun makePrediction(image: Bitmap) {
        binding.imageView.setImageBitmap(image)
        val modelName = binding.spinner.selectedItem.toString()
        viewModel.makePrediction(image, modelName)
    }

    companion object {
        fun getInstance() = DiseaseRecognitionFragment()
    }

}
