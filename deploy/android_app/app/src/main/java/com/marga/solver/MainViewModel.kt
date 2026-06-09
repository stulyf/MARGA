package com.marga.solver

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray

data class DatasetType(
    val depot: FloatArray,
    val loc: Array<FloatArray>,
    val prizes: FloatArray,
    val maxLength: Float
)

data class SolverHistory(
    val timestamp: Long,
    val result: MARGASolver.SolverResult,
    val nodeCount: Int
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val solver = MARGASolver(application)

    private val _dataset = MutableStateFlow<DatasetType?>(null)
    val dataset: StateFlow<DatasetType?> = _dataset

    private val _isSolving = MutableStateFlow(false)
    val isSolving: StateFlow<Boolean> = _isSolving

    private val _lastResult = MutableStateFlow<MARGASolver.SolverResult?>(null)
    val lastResult: StateFlow<MARGASolver.SolverResult?> = _lastResult

    private val _history = MutableStateFlow<List<SolverHistory>>(emptyList())
    val history: StateFlow<List<SolverHistory>> = _history

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { solver.load() }
        }
    }

    fun setDataset(data: DatasetType) {
        _dataset.value = data
        _lastResult.value = null
    }

    fun solve() {
        val data = _dataset.value ?: return
        viewModelScope.launch {
            _isSolving.value = true
            val result = withContext(Dispatchers.IO) {
                solver.solve(data.depot, data.loc, data.prizes, data.maxLength)
            }
            _lastResult.value = result
            _isSolving.value = false
            _history.value = (listOf(
                SolverHistory(System.currentTimeMillis(), result, data.loc.size)
            ) + _history.value).take(10)
        }
    }

    fun runBenchmark(onComplete: (Long, Long, Long) -> Unit) {
        val data = _dataset.value ?: return
        viewModelScope.launch {
            _isSolving.value = true
            val latencies = mutableListOf<Long>()
            withContext(Dispatchers.IO) {
                repeat(10) {
                    latencies.add(
                        solver.solve(data.depot, data.loc, data.prizes, data.maxLength).inferenceTimeMs
                    )
                }
            }
            _isSolving.value = false
            onComplete(
                latencies.average().toLong(),
                latencies.minOrNull() ?: 0,
                latencies.maxOrNull() ?: 0
            )
        }
    }

    /**
     * Load first instance from a bundled synthetic asset JSON and set as current dataset.
     */
    fun loadAssetDataset(filename: String) {
        viewModelScope.launch {
            val instance = withContext(Dispatchers.IO) {
                val json = getApplication<Application>().assets
                    .open(filename).bufferedReader().readText()
                val arr = JSONArray(json)
                if (arr.length() == 0) return@withContext null
                val obj = arr.getJSONObject(0)

                val depotArr = obj.getJSONArray("depot")
                val depot = floatArrayOf(
                    depotArr.getDouble(0).toFloat(),
                    depotArr.getDouble(1).toFloat()
                )
                val locArr = obj.getJSONArray("loc")
                val loc = Array(locArr.length()) { i ->
                    val row = locArr.getJSONArray(i)
                    floatArrayOf(row.getDouble(0).toFloat(), row.getDouble(1).toFloat())
                }
                val prizeArr = obj.getJSONArray("prize")
                val prizes = FloatArray(prizeArr.length()) { i -> prizeArr.getDouble(i).toFloat() }
                DatasetType(depot, loc, prizes, obj.getDouble("max_length").toFloat())
            }
            if (instance != null) setDataset(instance)
        }
    }

    override fun onCleared() {
        super.onCleared()
        solver.close()
    }
}
