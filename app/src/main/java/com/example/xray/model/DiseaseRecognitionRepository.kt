package com.example.xray.model

import java.nio.MappedByteBuffer

interface DiseaseRecognitionRepository {

    fun loadModelFile(modelName: String): MappedByteBuffer
}