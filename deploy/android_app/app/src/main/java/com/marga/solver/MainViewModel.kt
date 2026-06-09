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

sealed class DatasetType {
    data class Synthetic(
        val depot: FloatArray,
        val loc: Array<FloatArray>,
        val prizes: FloatArray,
        val maxLength: Float
    ) : DatasetType()

    data class RealWorld(
        val city: String,
        val depot: DoubleArray, // lon, lat
        val loc: Array<DoubleArray>,
        val prizes: FloatArray,
        val maxLength: Float
    ) : DatasetType()
}

data class SolverHistory(
    val timestamp: Long,
    val result: MARGASolver.SolverResult,
    val datasetType: String,
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
            withContext(Dispatchers.IO) {
                solver.load()
            }
        }
    }

    fun setDataset(data: DatasetType) {
        _dataset.value = data
        _lastResult.value = null
    }

    fun solve() {
        val currentData = _dataset.value ?: return
        viewModelScope.launch {
            _isSolving.value = true
            
            val result = withContext(Dispatchers.IO) {
                runInference(currentData)
            }
            
            _lastResult.value = result
            _isSolving.value = false
            
            addToHistory(result, currentData)
        }
    }

    fun runBenchmark(onComplete: (Long, Long, Long) -> Unit) {
        val currentData = _dataset.value ?: return
        viewModelScope.launch {
            _isSolving.value = true
            val latencies = mutableListOf<Long>()
            withContext(Dispatchers.IO) {
                repeat(10) {
                    val res = runInference(currentData)
                    latencies.add(res.inferenceTimeMs)
                }
            }
            _isSolving.value = false
            onComplete(latencies.average().toLong(), latencies.minOrNull() ?: 0, latencies.maxOrNull() ?: 0)
        }
    }

    private suspend fun runInference(data: DatasetType): MARGASolver.SolverResult {
        return when (data) {
            is DatasetType.Synthetic -> {
                solver.solve(data.depot, data.loc, data.prizes, data.maxLength)
            }
            is DatasetType.RealWorld -> {
                // Normalize for model
                val allCoords = listOf(data.depot) + data.loc
                val minLat = allCoords.minOf { it[1] }
                val maxLat = allCoords.maxOf { it[1] }
                val minLon = allCoords.minOf { it[0] }
                val maxLon = allCoords.maxOf { it[0] }
                
                val normDepot = floatArrayOf(
                    ((data.depot[0] - minLon) / (maxLon - minLon)).toFloat(),
                    ((data.depot[1] - minLat) / (maxLat - minLat)).toFloat()
                )
                val normLoc = Array(data.loc.size) { i ->
                    floatArrayOf(
                        ((data.loc[i][0] - minLon) / (maxLon - minLon)).toFloat(),
                        ((data.loc[i][1] - minLat) / (maxLat - minLat)).toFloat()
                    )
                }
                solver.solve(normDepot, normLoc, data.prizes, data.maxLength)
            }
        }
    }

    private fun addToHistory(result: MARGASolver.SolverResult, data: DatasetType) {
        val historyEntry = SolverHistory(
            timestamp = System.currentTimeMillis(),
            result = result,
            datasetType = if (data is DatasetType.Synthetic) "Synthetic" else "RealWorld",
            nodeCount = when (data) {
                is DatasetType.Synthetic -> data.loc.size
                is DatasetType.RealWorld -> data.loc.size
            }
        )
        _history.value = (listOf(historyEntry) + _history.value).take(10)
    }

    /**
     * Load first instance from a bundled asset JSON dataset and set it as current.
     * File format: JSON array of instances with depot, loc, prize, max_length fields.
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
                val maxLength = obj.getDouble("max_length").toFloat()
                val type = if (obj.has("type")) obj.getString("type") else "synthetic"
                val city = if (obj.has("city")) obj.getString("city") else null

                if (type == "realworld" && city != null) {
                    // depot and loc are lon/lat doubles for real-world
                    DatasetType.RealWorld(
                        city = city,
                        depot = doubleArrayOf(depot[0].toDouble(), depot[1].toDouble()),
                        loc = Array(loc.size) { i ->
                            doubleArrayOf(loc[i][0].toDouble(), loc[i][1].toDouble())
                        },
                        prizes = prizes,
                        maxLength = maxLength
                    )
                } else {
                    DatasetType.Synthetic(depot, loc, prizes, maxLength)
                }
            }
            if (instance != null) setDataset(instance)
        }
    }

    override fun onCleared() {
        super.onCleared()
        solver.close()
    }
}
