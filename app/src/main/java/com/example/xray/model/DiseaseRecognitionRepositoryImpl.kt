package com.example.xray.model

import android.content.Context
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DiseaseRecognitionRepositoryImpl(private val context: Context): DiseaseRecognitionRepository {

    @Throws(IOException::class)
    override fun loadModelFile(modelName: String): MappedByteBuffer {
        val filename = "models/$modelName"
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

}