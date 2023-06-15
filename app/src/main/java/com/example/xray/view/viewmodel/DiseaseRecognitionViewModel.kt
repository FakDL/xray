package com.example.xray.view.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import androidx.lifecycle.AndroidViewModel
import com.example.xray.model.DiseaseRecognitionRepositoryImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class DiseaseRecognitionViewModel(application: Application) : AndroidViewModel(application) {

    private val _predictedLabel: MutableStateFlow<String> = MutableStateFlow(EMPTY_STRING)
    val predictedLabel = _predictedLabel.asStateFlow()

    private val repository = DiseaseRecognitionRepositoryImpl(application)

    fun makePrediction(image: Bitmap, modelName: String) {
        val compressedImage = compressBitmap(image)
        val loadedModel = repository.loadModelFile(modelName)
        val model = Interpreter(loadedModel)
        val inputImageBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), DataType.FLOAT32)
        val arr = convertBitmapToFloatArray(compressedImage)
        val outputTensor = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
        preprocessImage(arr).also { inputImageBuffer.loadArray(it) }
        model.run(inputImageBuffer.buffer, outputTensor.buffer.rewind())

        val probabilities = outputTensor.floatArray

        val index = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        val classes = arrayOf("COVID-19", "болезнь не найдена", "пневмония", "пневмония")
        val predictedLabel = classes[index]

        _predictedLabel.value = predictedLabel

        model.close()
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



    private fun preprocessImage(arr: FloatArray): FloatArray {

        for (i in arr.indices step 3) {
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



    companion object {
        const val IMAGE_SIZE = 300
        const val CHANNELS = 3
        const val EMPTY_STRING = ""
    }

}