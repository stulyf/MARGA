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
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

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
                DatasetType.Synthetic(floatArrayOf(0.5f, 0.5f), emptyArray(), floatArrayOf(), 2.0f)
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
            val cur = viewModel.dataset.value as? DatasetType.Synthetic
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
            val cur = viewModel.dataset.value as? DatasetType.Synthetic
            if (cur != null) viewModel.setDataset(cur.copy(depot = floatArrayOf(x, y)))
        }

        // ── Observe dataset ──────────────────────────────────────────────────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.dataset.collectLatest { data ->
                when (data) {
                    is DatasetType.Synthetic -> {
                        binding.canvasView.visibility = View.VISIBLE
                        binding.mapView.visibility    = View.GONE
                        binding.canvasView.setData(data, viewModel.lastResult.value)
                        updateNodeCountChip(data.loc.size)
                    }
                    is DatasetType.RealWorld -> {
                        binding.canvasView.visibility = View.GONE
                        binding.mapView.visibility    = View.VISIBLE
                        setupMapView(data, viewModel.lastResult.value)
                        updateNodeCountChip(data.loc.size)
                    }
                    null -> updateNodeCountChip(0)
                }
            }
        }

        // ── Observe result ───────────────────────────────────────────────────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.lastResult.collectLatest { result ->
                if (result != null) {
                    (viewModel.dataset.value as? DatasetType.Synthetic)?.let { data ->
                        binding.canvasView.setData(data, result)
                    }
                    updateResultsUI(result)
                    bottomSheet.state = BottomSheetBehavior.STATE_COLLAPSED
                }
            }
        }

        // ── Observe solving state (disable Run button, show progress) ──────
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.isSolving.collectLatest { solving ->
                binding.solveProgress.visibility = if (solving) View.VISIBLE else View.GONE
                binding.btnRun.isEnabled = !solving
                binding.btnRun.text = if (solving) "Running…" else "Run"
            }
        }

        if (viewModel.dataset.value == null) generateRandomSynthetic()
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun updateNodeCountChip(count: Int) {
        binding.chipNodeCount.text = "$count nodes"
    }

    private fun generateRandomSynthetic() {
        val rng = Random(System.currentTimeMillis())
        viewModel.setDataset(
            DatasetType.Synthetic(
                depot     = floatArrayOf(rng.nextFloat(), rng.nextFloat()),
                loc       = Array(100) { floatArrayOf(rng.nextFloat(), rng.nextFloat()) },
                prizes    = FloatArray(100) { 1f },
                maxLength = 2.0f
            )
        )
    }

    private fun updateResultsUI(result: MARGASolver.SolverResult) {
        val data = viewModel.dataset.value
        val totalNodes = when (data) {
            is DatasetType.Synthetic -> data.loc.size
            is DatasetType.RealWorld -> data.loc.size
            else -> 0
        }
        val visitedCount = result.routes.sumOf { r -> r.count { it in 1..totalNodes } }

        binding.bottomSheetResults.tvInferenceTime.text = "${result.inferenceTimeMs} ms"
        binding.bottomSheetResults.tvTotalReward.text = "%.0f".format(result.totalPrize)
        binding.bottomSheetResults.tvNodesVisited.text =
            "Nodes visited: $visitedCount / $totalNodes"
    }

    // ── Map view ──────────────────────────────────────────────────────────────

    private fun setupMapView(data: DatasetType.RealWorld, result: MARGASolver.SolverResult?) {
        binding.mapView.setTileSource(TileSourceFactory.MAPNIK)
        binding.mapView.setMultiTouchControls(true)
        binding.mapView.controller.setZoom(15.0)
        val startPoint = GeoPoint(data.depot[1], data.depot[0])
        binding.mapView.controller.setCenter(startPoint)
        binding.mapView.overlays.clear()

        binding.mapView.overlays.add(Marker(binding.mapView).apply {
            position = startPoint
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
            title = "Depot"
        })
        data.loc.forEachIndexed { i, loc ->
            binding.mapView.overlays.add(Marker(binding.mapView).apply {
                position = GeoPoint(loc[1], loc[0])
                title = "Node $i"
            })
        }

        val agentColors = listOf(
            android.graphics.Color.RED, android.graphics.Color.BLUE,
            android.graphics.Color.GREEN, android.graphics.Color.parseColor("#FF9800"),
            android.graphics.Color.MAGENTA
        )
        result?.routes?.forEachIndexed { agentIdx, route ->
            val polyline = Polyline().apply {
                outlinePaint.color = agentColors[agentIdx % agentColors.size]
                outlinePaint.strokeWidth = 5f
            }
            val pts = mutableListOf(startPoint)
            route.forEach { nodeIdx ->
                if (nodeIdx in 1..data.loc.size)
                    pts.add(GeoPoint(data.loc[nodeIdx - 1][1], data.loc[nodeIdx - 1][0]))
            }
            pts.add(startPoint)
            polyline.setPoints(pts)
            binding.mapView.overlays.add(polyline)
        }
        binding.mapView.invalidate()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
