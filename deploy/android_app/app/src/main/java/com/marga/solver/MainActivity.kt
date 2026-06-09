package com.marga.solver

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.marga.solver.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_plan     -> showFragment(PlanFragment(),     "plan")
                R.id.nav_results  -> showFragment(ResultsFragment(),  "results")
                R.id.nav_settings -> showFragment(SettingsFragment(), "settings")
            }
            true
        }

        if (savedInstanceState == null) {
            binding.bottomNav.selectedItemId = R.id.nav_plan
        } else {
            // Restore the previously selected tab
            val tag = savedInstanceState.getString(KEY_CURRENT_TAG, "plan")
            val fragment = supportFragmentManager.findFragmentByTag(tag)
                ?: fragmentForTag(tag)
            showFragment(fragment, tag)
            binding.bottomNav.selectedItemId = menuIdForTag(tag)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        val current = supportFragmentManager.findFragmentById(R.id.navHostFragment)
        outState.putString(KEY_CURRENT_TAG, current?.tag ?: "plan")
    }

    private fun showFragment(fragment: Fragment, tag: String) {
        val existing = supportFragmentManager.findFragmentByTag(tag)
        supportFragmentManager.beginTransaction()
            .replace(R.id.navHostFragment, existing ?: fragment, tag)
            .commit()
    }

    private fun fragmentForTag(tag: String): Fragment = when (tag) {
        "results"  -> ResultsFragment()
        "settings" -> SettingsFragment()
        else       -> PlanFragment()
    }

    private fun menuIdForTag(tag: String): Int = when (tag) {
        "results"  -> R.id.nav_results
        "settings" -> R.id.nav_settings
        else       -> R.id.nav_plan
    }

    companion object {
        private const val KEY_CURRENT_TAG = "current_tab"
    }
}
