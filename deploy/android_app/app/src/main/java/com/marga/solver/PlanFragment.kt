package com.marga.solver

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.marga.solver.databinding.FragmentPlanBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.random.Random

class PlanFragment : Fragment() {

    private var _binding: FragmentPlanBinding? = null
    private val binding get() = _binding!!
    private val viewModel: MainViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentPlanBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val bottomSheet = BottomSheetBehavior.from(binding.bottomSheetResults.resultsBottomSheet)
        bottomSheet.state = BottomSheetBehavior.STATE_HIDDEN

        // ── Toolbar ─────────────────────────────────────────────────────────
        binding.btnRandom.setOnClickListener { generateRandomSynthetic() }
        binding.btnClear.setOnClickListener {
            viewModel.setDataset(
                DatasetType(floatArrayOf(0.5f, 0.5f), emptyArray(), floatArrayOf(), 2.0f)
            )
            bottomSheet.state = BottomSheetBehavior.STATE_HIDDEN
        }
        binding.btnRun.setOnClickListener { viewModel.solve() }

        // ── Bottom-sheet actions ─────────────────────────────────────────────
        binding.bottomSheetResults.btnRunAgain.setOnClickListener { viewModel.solve() }
        binding.bottomSheetResults.btnBenchmark.setOnClickListener {
            viewModel.runBenchmark { avg, min, max ->
                BenchmarkDialog().apply {
                    this.avg = avg; this.min = min; this.max = max
                }.show(parentFragmentManager, "benchmark")
            }
        }

        // ── Canvas interaction ───────────────────────────────────────────────
        binding.canvasView.onNodeAdded = { x, y ->
            val cur = viewModel.dataset.value
            if (cur != null) {
                viewModel.setDataset(
                    cur.copy(
                        loc    = cur.loc + arrayOf(floatArrayOf(x, y)),
                        prizes = FloatArray(cur.prizes.size + 1).also { a ->
                            cur.prizes.copyInto(a); a[a.size - 1] = 1f
                        }
                    )
                )
            }
        }
        binding.canvasView.onDepotSet = { x, y ->
            viewModel.dataset.value?.let {
                viewModel.setDataset(it.copy(depot = floatArrayOf(x, y)))
            }
        }

        // ── Observe dataset ──────────────────────────────────────────────────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.dataset.collectLatest { data ->
                if (data != null) {
                    binding.canvasView.setData(data, viewModel.lastResult.value)
                    binding.chipNodeCount.text = "${data.loc.size} nodes"
                } else {
                    binding.chipNodeCount.text = "0 nodes"
                }
            }
        }

        // ── Observe result ───────────────────────────────────────────────────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.lastResult.collectLatest { result ->
                if (result != null) {
                    viewModel.dataset.value?.let { data ->
                        binding.canvasView.setData(data, result)
                    }
                    updateResultsUI(result)
                    bottomSheet.state = BottomSheetBehavior.STATE_COLLAPSED
                }
            }
        }

        // ── Observe solving state ────────────────────────────────────────────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.isSolving.collectLatest { solving ->
                binding.solveProgress.visibility = if (solving) View.VISIBLE else View.GONE
                binding.btnRun.isEnabled = !solving
                binding.btnRun.text = if (solving) "Running…" else "Run"
            }
        }

        if (viewModel.dataset.value == null) generateRandomSynthetic()
    }

    private fun generateRandomSynthetic() {
        val rng = Random(System.currentTimeMillis())
        viewModel.setDataset(
            DatasetType(
                depot     = floatArrayOf(rng.nextFloat(), rng.nextFloat()),
                loc       = Array(100) { floatArrayOf(rng.nextFloat(), rng.nextFloat()) },
                prizes    = FloatArray(100) { 1f },
                maxLength = 2.0f
            )
        )
    }

    private fun updateResultsUI(result: MARGASolver.SolverResult) {
        val totalNodes = viewModel.dataset.value?.loc?.size ?: 0
        val visitedCount = result.routes.sumOf { r -> r.count { it in 1..totalNodes } }
        binding.bottomSheetResults.tvInferenceTime.text = "${result.inferenceTimeMs} ms"
        binding.bottomSheetResults.tvTotalReward.text = "%.0f".format(result.totalPrize)
        binding.bottomSheetResults.tvNodesVisited.text = "Nodes visited: $visitedCount / $totalNodes"
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
