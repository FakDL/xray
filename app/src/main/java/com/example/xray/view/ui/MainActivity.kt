package com.example.xray.view.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.example.xray.R
import com.example.xray.databinding.ActivityMainBinding


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        toFragment(DiseaseRecognitionFragment.getInstance())
    }

    private fun toFragment(fragment: Fragment) {
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.am_container, fragment)
            .commit()
    }

}
