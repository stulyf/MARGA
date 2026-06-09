package com.marga.solver

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import com.google.gson.Gson
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * MARGA Solver — ONNX Runtime Android inference
 *
 * Model split:
 *   encoder.onnx       — runs once per problem instance
 *   decoder_step.onnx  — runs once per decoding step
 *
 * Java-side control flow:
 *   - Greedy / stochastic node selection
 *   - Entropy scheduling (highest-entropy agent decoded first)
 *   - Feasibility mask (visited + route-length budget)
 *   - State update (visited mask, lengths, current_node)
 */
class MARGASolver(private val context: Context) {

    companion object {
        private const val TAG = "MARGASolver"
        // End-depot is always the last node (index = N+1)
        private const val MAX_STEPS = 300
        // Temperature for sampling (1.0 = greedy argmax via temp→0 softmax)
        private const val TEMPERATURE = 1.0f
    }

    data class Config(
        val n_nodes: Int,
        val num_agents: Int,
        val embed_dim: Int,
        val n_heads: Int,
        val tanh_clip: Float,
        val n2: Int,       // n_nodes + 2
    )

    data class SolverResult(
        val routes: List<List<Int>>,    // per-agent node visit sequence (0-indexed)
        val totalPrize: Float,
        val inferenceTimeMs: Long,
    )

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var encSession: OrtSession
    private lateinit var decSession: OrtSession
    private lateinit var config: Config

    fun load() {
        val t0 = System.currentTimeMillis()
        ortEnv = OrtEnvironment.getEnvironment()

        // Session options: use NNAPI if available, else CPU
        val opts = OrtSession.SessionOptions()
        try {
            opts.addNnapi()
            Log.i(TAG, "NNAPI execution provider enabled")
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI not available, using CPU")
        }
        opts.setIntraOpNumThreads(4)

        encSession = loadSession("encoder.onnx", opts)
        decSession = loadSession("decoder_step.onnx", opts)
        config     = loadConfig()

        Log.i(TAG, "Models loaded in ${System.currentTimeMillis() - t0} ms  " +
                   "n=${config.n_nodes} K=${config.num_agents}")
    }

    private fun loadSession(filename: String, opts: OrtSession.SessionOptions): OrtSession {
        val bytes = context.assets.open(filename).readBytes()
        return ortEnv.createSession(bytes, opts)
    }

    private fun loadConfig(): Config {
        val json = context.assets.open("config.json").bufferedReader().readText()
        return Gson().fromJson(json, Config::class.java)
    }

    /**
     * Solve a single routing problem instance.
     *
     * @param depotXY     depot coordinates [2]
     * @param nodeXY      delivery node coordinates [N, 2], values in [0,1]
     * @param prizes      prize per node [N], typically uniform (all 1.0)
     * @param maxLength   maximum total route length (e.g. 2.0)
     * @param stochastic  if true, sample from distribution; else argmax
     */
    fun solve(
        depotXY: FloatArray,
        nodeXY: Array<FloatArray>,
        prizes: FloatArray,
        maxLength: Float,
        stochastic: Boolean = false,
    ): SolverResult {
        val t0 = System.currentTimeMillis()
        val N  = nodeXY.size
        val N2 = N + 2
        val K  = config.num_agents

        // ── 1. Encode ─────────────────────────────────────────────────────
        val (nodeEmb, fixedCtx, gKey, gVal, logitKey, nodeMaxLen, coords) =
            runEncoder(depotXY, nodeXY, prizes, maxLength, N, N2)

        // ── 2. Build coords from encoder output ────────────────────────────
        // coords are now returned directly by encoder.onnx
        val prizeArr = FloatArray(N2).also { p ->
            for (i in 1..N) p[i] = prizes[i - 1]
        }

        // ── 3. Initialize state ────────────────────────────────────────────
        val currentNode = IntArray(K) { 0 }          // all start at depot (idx=0)
        val lengths     = FloatArray(K) { 0f }
        val visited     = BooleanArray(N2) { false }
        visited[0]    = true   // start depot
        visited[N2-1] = true   // end depot (will be freed later when needed)

        val routes = Array(K) { mutableListOf<Int>() }
        val done   = BooleanArray(K) { false }
        var totalPrize = 0f

        // ── 4. Decoding loop ───────────────────────────────────────────────
        repeat(MAX_STEPS) { _ ->
            if (done.all { it }) return@repeat

            // Run decoder step once — gets log_p for ALL agents simultaneously
            val logP = runDecoderStep(
                nodeEmb, fixedCtx, gKey, gVal, logitKey, nodeMaxLen, coords,
                currentNode, lengths, visited, N2, K
            )
            // logP: [K, N2]

            // ── Entropy scheduling: LOW entropy (most certain) agent goes FIRST
            //    (matches _compute_entropy_priority: argsort ascending)
            val agentOrder = computeEntropyOrder(logP, K, N2)

            // ── Intra-step progressive masking (matches _select_nodes_dynamic_order)
            //    After agent k selects node n, remove n from subsequent agents' logP
            val logPAux = Array(K) { k -> logP[k].copyOf() }
            val selections = IntArray(K) { -1 }

            for (ki in agentOrder.indices) {
                val k = agentOrder[ki]
                if (done[k]) continue

                val sel = if (stochastic) sampleFromLogP(logPAux[k])
                          else           argmax(logPAux[k])
                selections[k] = sel

                // Mask this node for all remaining agents (end depot always allowed)
                if (sel != N2 - 1) {
                    for (j in ki + 1 until agentOrder.size) {
                        val k2 = agentOrder[j]
                        if (done[k2]) continue
                        val endVal = logPAux[k2][N2 - 1]
                        logPAux[k2][sel] = -1e9f
                        // Re-normalize via log-softmax — use Double to match numpy precision
                        val maxV = logPAux[k2].fold(Double.NEGATIVE_INFINITY) { acc, v ->
                            if (v > -1e8f) maxOf(acc, v.toDouble()) else acc
                        }
                        var sumExp = 0.0
                        for (n in 0 until N2) {
                            if (logPAux[k2][n] > -1e8f)
                                sumExp += Math.exp(logPAux[k2][n].toDouble() - maxV)
                        }
                        for (n in 0 until N2) {
                            logPAux[k2][n] = if (logPAux[k2][n] > -1e8f)
                                Math.log(Math.exp(logPAux[k2][n].toDouble() - maxV) / sumExp + 1e-9).toFloat()
                            else -1e9f
                        }
                        logPAux[k2][N2 - 1] = endVal  // restore end depot
                    }
                }
            }

            // ── Update state for all agents simultaneously
            for (k in 0 until K) {
                val sel = selections[k]
                if (done[k] || sel == -1) continue

                val dist = euclidean(coords[currentNode[k]], coords[sel])
                lengths[k]     += dist
                currentNode[k]  = sel
                routes[k].add(sel)
                if (sel != N2 - 1) {
                    totalPrize  += prizeArr[sel]
                    visited[sel] = true
                }
                if (sel == N2 - 1) done[k] = true
            }
        }

        val elapsed = System.currentTimeMillis() - t0
        return SolverResult(
            routes = routes.map { it.toList() },
            totalPrize = totalPrize,
            inferenceTimeMs = elapsed,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ONNX inference helpers
    // ─────────────────────────────────────────────────────────────────────────

    private data class EncoderOutputs(
        val nodeEmb: Array<FloatArray>,
        val fixedCtx: FloatArray,
        val glimpseKey: Array<Array<Array<Array<FloatArray>>>>,
        val glimpseVal: Array<Array<Array<Array<FloatArray>>>>,
        val logitKey: Array<FloatArray>,
        val nodeMaxLen: FloatArray,
        val coords: Array<FloatArray>,    // [N2, 2]  ← NEW
    )

    private fun runEncoder(
        depot: FloatArray, loc: Array<FloatArray>,
        prize: FloatArray, maxLen: Float,
        N: Int, N2: Int,
    ): EncoderOutputs {
        val D   = config.embed_dim
        val H   = config.n_heads
        val dh  = D / H

        // Flatten inputs
        val depotBuf  = FloatBuffer.wrap(depot)
        val locFlat   = FloatArray(N * 2) { i -> loc[i / 2][i % 2] }
        val locBuf    = FloatBuffer.wrap(locFlat)
        val prizeBuf  = FloatBuffer.wrap(prize)
        val maxLenBuf = FloatBuffer.wrap(floatArrayOf(maxLen))

        val inputs = mapOf(
            "depot"      to OnnxTensor.createTensor(ortEnv, depotBuf,  longArrayOf(1, 2)),
            "loc"        to OnnxTensor.createTensor(ortEnv, locBuf,    longArrayOf(1, N.toLong(), 2)),
            "prize"      to OnnxTensor.createTensor(ortEnv, prizeBuf,  longArrayOf(1, N.toLong())),
            "max_length" to OnnxTensor.createTensor(ortEnv, maxLenBuf, longArrayOf(1)),
        )

        val result = encSession.run(inputs)

        val nodeEmbFlat  = (result[0].value as Array<*>).flattenToFloatArray()
        val fixedCtxFlat = (result[1].value as Array<*>).flattenToFloatArray()
        val gKeyFlat     = (result[2].value as Array<*>).flattenToFloatArray()
        val gValFlat     = (result[3].value as Array<*>).flattenToFloatArray()
        val logitKeyFlat = (result[4].value as Array<*>).flattenToFloatArray()
        val nmlFlat      = (result[5].value as Array<*>).flattenToFloatArray()
        val coordsFlat   = (result[6].value as Array<*>).flattenToFloatArray()  // [1, N2, 2]

        val nodeEmb    = Array(N2) { n -> FloatArray(D) { d -> nodeEmbFlat[n * D + d] } }
        val fixedCtx   = FloatArray(D) { d -> fixedCtxFlat[d] }
        val gKey = Array(H) { h -> Array(N2) { n ->
            Array(1) { Array(1) { FloatArray(dh) { d -> gKeyFlat[((h * N2 + n) * dh) + d] } } }
        }}
        val gVal = Array(H) { h -> Array(N2) { n ->
            Array(1) { Array(1) { FloatArray(dh) { d -> gValFlat[((h * N2 + n) * dh) + d] } } }
        }}
        val logitKey   = Array(N2) { n -> FloatArray(D) { d -> logitKeyFlat[n * D + d] } }
        val nodeMaxLen = FloatArray(N2) { n -> nmlFlat[n] }
        val coords     = Array(N2) { n -> FloatArray(2) { i -> coordsFlat[n * 2 + i] } }  // NEW

        inputs.values.forEach { it.close() }
        result.close()

        return EncoderOutputs(nodeEmb, fixedCtx, gKey, gVal, logitKey, nodeMaxLen, coords)
    }

    private fun runDecoderStep(
        nodeEmb: Array<FloatArray>,
        fixedCtx: FloatArray,
        gKey: Array<Array<Array<Array<FloatArray>>>>,
        gVal: Array<Array<Array<Array<FloatArray>>>>,
        logitKey: Array<FloatArray>,
        nodeMaxLen: FloatArray,
        coords: Array<FloatArray>,    // NEW
        currentNode: IntArray,
        lengths: FloatArray,
        visited: BooleanArray,
        N2: Int, K: Int,
    ): Array<FloatArray> {
        val D  = config.embed_dim
        val H  = config.n_heads
        val dh = D / H

        // Flatten tensors to match expected ONNX shapes
        // node_embeddings: [1, N2, D]
        val neFlat = FloatArray(N2 * D) { i -> nodeEmb[i / D][i % D] }
        // fixed_context: [1, 1, D]
        val fcFlat = FloatArray(D) { i -> fixedCtx[i] }
        // glimpse_key: [H, 1, 1, N2, dh]
        val gkFlat = FloatArray(H * N2 * dh) { i ->
            val h = i / (N2 * dh); val n = (i / dh) % N2; val d = i % dh
            gKey[h][n][0][0][d]
        }
        val gvFlat = FloatArray(H * N2 * dh) { i ->
            val h = i / (N2 * dh); val n = (i / dh) % N2; val d = i % dh
            gVal[h][n][0][0][d]
        }
        // logit_key: [1, 1, N2, D]
        val lkFlat = FloatArray(N2 * D) { i -> logitKey[i / D][i % D] }
        // current_node: [1, K]
        val cnFlat = LongArray(K) { k -> currentNode[k].toLong() }
        // lengths: [1, K]
        val lenFlat = FloatArray(K) { k -> lengths[k] }
        // visited: [1, N2] uint8
        val visFlat = ByteArray(N2) { n -> if (visited[n]) 1 else 0 }

        // coords: [1, N2, 2]
        val coordsFlat = FloatArray(N2 * 2) { i -> coords[i / 2][i % 2] }

        val inputs = mapOf(
            "node_embeddings" to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(neFlat),
                                    longArrayOf(1, N2.toLong(), D.toLong())),
            "fixed_context"   to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(fcFlat),
                                    longArrayOf(1, 1, D.toLong())),
            "glimpse_key"     to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(gkFlat),
                                    longArrayOf(H.toLong(), 1, 1, N2.toLong(), dh.toLong())),
            "glimpse_val"     to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(gvFlat),
                                    longArrayOf(H.toLong(), 1, 1, N2.toLong(), dh.toLong())),
            "logit_key"       to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(lkFlat),
                                    longArrayOf(1, 1, N2.toLong(), D.toLong())),
            "node_max_length" to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(nodeMaxLen),
                                    longArrayOf(1, N2.toLong())),
            "coords"          to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(coordsFlat),
                                    longArrayOf(1, N2.toLong(), 2)),
            "current_node"    to OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(cnFlat),
                                    longArrayOf(1, K.toLong())),
            "lengths"         to OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(lenFlat),
                                    longArrayOf(1, K.toLong())),
            "visited"         to OnnxTensor.createTensor(ortEnv, ByteBuffer.wrap(visFlat),
                                    longArrayOf(1, N2.toLong()), OnnxJavaType.UINT8),
        )

        val result  = decSession.run(inputs)
        val logpFlat = (result[0].value as Array<*>).flattenToFloatArray()  // [1, K, N2]

        inputs.values.forEach { it.close() }
        result.close()

        // Reshape to [K, N2]
        return Array(K) { k -> FloatArray(N2) { n -> logpFlat[k * N2 + n] } }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State management helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Compute per-agent entropy, return agent indices sorted LOW-entropy first (most certain first).
     *  Uses Double precision throughout to match numpy float64 behavior. */
    private fun computeEntropyOrder(logP: Array<FloatArray>, K: Int, N2: Int): IntArray {
        val entropy = DoubleArray(K) { k ->
            // Sum of exp(logP) for normalization — stay in Double
            var sumExp = 0.0
            for (n in 0 until N2) {
                val e = Math.exp(logP[k][n].toDouble())
                if (e > 1e-9) sumExp += e
            }
            // H = -Σ p*log(p)
            var h = 0.0
            for (n in 0 until N2) {
                val p = Math.exp(logP[k][n].toDouble()) / (sumExp + 1e-9)
                if (p > 1e-9) h -= p * Math.log(p)
            }
            h
        }
        // ascending: low entropy (certain) first
        return (0 until K).sortedBy { entropy[it] }.toIntArray()
    }

    /**
     * Apply feasibility mask: forbid visiting already-visited nodes and
     * nodes that would exceed the route length budget.
     */
    private fun applyMask(
        logP: FloatArray,
        k: Int,
        currentNode: IntArray,
        lengths: FloatArray,
        nodeMaxLen: FloatArray,
        visited: BooleanArray,
        N2: Int,
    ): FloatArray {
        val masked = logP.copyOf()
        val endDepot = N2 - 1
        for (n in 0 until N2) {
            if (visited[n] && n != endDepot) {
                masked[n] = Float.NEGATIVE_INFINITY
            }
            // Feasibility: remaining budget check is handled by nodeMaxLen
            if (lengths[k] > nodeMaxLen[n]) {
                masked[n] = Float.NEGATIVE_INFINITY
            }
        }
        // Always allow end depot
        if (masked[endDepot] == Float.NEGATIVE_INFINITY) {
            masked[endDepot] = -1e4f
        }
        return masked
    }

    private fun argmax(arr: FloatArray): Int = arr.indices.maxByOrNull { arr[it] } ?: 0

    private fun sampleFromLogP(logP: FloatArray): Int {
        // Convert log-probs to probabilities, then sample
        val maxLp = logP.max() ?: 0f
        val probs = FloatArray(logP.size) { i ->
            if (logP[i] == Float.NEGATIVE_INFINITY) 0f
            else Math.exp((logP[i] - maxLp).toDouble()).toFloat()
        }
        val total = probs.sum()
        var r = Math.random().toFloat() * total
        for (i in probs.indices) {
            r -= probs[i]
            if (r <= 0f) return i
        }
        return probs.indices.last
    }

    private fun euclidean(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]
        return sqrt(dx * dx + dy * dy)
    }

    fun close() {
        encSession.close()
        decSession.close()
        ortEnv.close()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension: recursively flatten nested Arrays returned by ORT to FloatArray
// ─────────────────────────────────────────────────────────────────────────────
private fun Array<*>.flattenToFloatArray(): FloatArray {
    val result = mutableListOf<Float>()
    flattenInto(result)
    return result.toFloatArray()
}

private fun Any?.flattenInto(out: MutableList<Float>) {
    when (this) {
        is FloatArray -> this.forEach { out.add(it) }
        is Array<*>   -> this.forEach { it.flattenInto(out) }
        is Float      -> out.add(this)
        else          -> {}
    }
}
