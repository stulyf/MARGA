package com.marga.solver

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.marga.solver.databinding.FragmentSettingsBinding
import kotlinx.coroutines.launch

class SettingsFragment : Fragment() {

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!
    private val viewModel: MainViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // ── Dataset buttons ─────────────────────────────────────────────────
        binding.btnLoadSynthConst.setOnClickListener {
            loadDataset("synthetic_const_n100_mini.json", "Synthetic Const")
        }
        binding.btnLoadSynthUnif.setOnClickListener {
            loadDataset("synthetic_unif_n100_mini.json", "Synthetic Uniform")
        }
        binding.btnLoadSynthDist.setOnClickListener {
            loadDataset("synthetic_dist_n100_mini.json", "Synthetic Dist")
        }

        // ── Benchmark ───────────────────────────────────────────────────────
        binding.btnRunBenchmark.setOnClickListener {
            if (viewModel.dataset.value == null) {
                Toast.makeText(context, "Please load a dataset first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            binding.btnRunBenchmark.isEnabled = false
            binding.btnRunBenchmark.text = "Running…"
            viewModel.runBenchmark { avg, min, max ->
                binding.cardBenchResult.visibility = View.VISIBLE
                binding.tvBenchAvg.text = "Avg: $avg ms"
                binding.tvBenchMinMax.text = "Min: $min ms  Max: $max ms"
                binding.btnRunBenchmark.isEnabled = true
                binding.btnRunBenchmark.text = "Run Benchmark (10 runs)"
            }
        }
    }

    private fun loadDataset(filename: String, label: String) {
        viewModel.loadAssetDataset(filename)
        Toast.makeText(context, "Loaded: $label", Toast.LENGTH_SHORT).show()
        // Jump to Plan tab so the user sees the loaded data immediately
        requireActivity().findViewById<BottomNavigationView>(R.id.bottomNav)
            ?.selectedItemId = R.id.nav_plan
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
