package com.marga.solver

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Canvas that renders the MARGA routing solution.
 * Draws all routes immediately (no step-by-step animation).
 * Visited nodes are colored by the agent that visited them.
 */
class MARGACanvasView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ── Paints ───────────────────────────────────────────────────────────────

    private val gridPaint = Paint().apply {
        color = Color.argb(60, 200, 200, 200)
        strokeWidth = 1f
        style = Paint.Style.STROKE
    }

    private val nodeUnvisitedPaint = Paint().apply {
        color = Color.parseColor("#BDBDBD")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val nodeOutlinePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }

    private val nodeFillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val depotFillPaint = Paint().apply {
        color = Color.parseColor("#FFD700")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val depotStrokePaint = Paint().apply {
        color = Color.parseColor("#F57F17")
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val depotLabelPaint = Paint().apply {
        color = Color.parseColor("#4E342E")
        textSize = 26f
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
        isAntiAlias = true
    }

    private val agentColors = listOf(
        R.color.agent_0, R.color.agent_1, R.color.agent_2,
        R.color.agent_3, R.color.agent_4
    ).map { ContextCompat.getColor(context, it) }

    private val routePaints = agentColors.map { color ->
        Paint().apply {
            this.color = color
            strokeWidth = 5f
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
        }
    }

    // ── State ─────────────────────────────────────────────────────────────────

    private var dataset: DatasetType.Synthetic? = null
    private var result: MARGASolver.SolverResult? = null
    /** nodeIdx (1-based) → agentIdx that visited it */
    private val nodeVisitedBy = mutableMapOf<Int, Int>()
    private val padding = 0.08f

    var onNodeAdded: ((Float, Float) -> Unit)? = null
    var onDepotSet:  ((Float, Float) -> Unit)? = null

    // ── Touch ─────────────────────────────────────────────────────────────────

    private val gestureDetector = GestureDetector(context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapUp(e: MotionEvent): Boolean {
                val nx = (e.x / width  - padding) / (1 - 2 * padding)
                val ny = (e.y / height - padding) / (1 - 2 * padding)
                if (nx in 0f..1f && ny in 0f..1f) onNodeAdded?.invoke(nx, ny)
                return true
            }
            override fun onLongPress(e: MotionEvent) {
                val nx = (e.x / width  - padding) / (1 - 2 * padding)
                val ny = (e.y / height - padding) / (1 - 2 * padding)
                if (nx in 0f..1f && ny in 0f..1f) onDepotSet?.invoke(nx, ny)
            }
        })

    override fun onTouchEvent(event: MotionEvent): Boolean =
        gestureDetector.onTouchEvent(event) || super.onTouchEvent(event)

    // ── Public API ────────────────────────────────────────────────────────────

    /** Update dataset and/or result, redraw immediately (no animation). */
    fun setData(dataset: DatasetType.Synthetic?, result: MARGASolver.SolverResult? = null) {
        this.dataset = dataset
        this.result  = result
        nodeVisitedBy.clear()
        result?.routes?.forEachIndexed { agentIdx, route ->
            route.forEach { nodeIdx -> nodeVisitedBy[nodeIdx] = agentIdx }
        }
        invalidate()
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawGrid(canvas)
        val data = dataset ?: return

        val depotX = tx(data.depot[0])
        val depotY = ty(data.depot[1])

        // ── Routes (drawn beneath nodes so nodes appear on top) ──────────────
        result?.routes?.forEachIndexed { agentIdx, route ->
            if (route.isEmpty()) return@forEachIndexed
            val paint = routePaints[agentIdx % routePaints.size]
            val path  = Path()
            var prevX = depotX; var prevY = depotY
            path.moveTo(prevX, prevY)

            route.forEach { nodeIdx ->
                val (cx, cy) = coordsFor(data, nodeIdx)
                path.lineTo(cx, cy)
                drawArrowIfLongEnough(canvas, prevX, prevY, cx, cy, paint)
                prevX = cx; prevY = cy
            }
            canvas.drawPath(path, paint)
        }

        // ── Delivery nodes ───────────────────────────────────────────────────
        data.loc.forEachIndexed { index, loc ->
            val x = tx(loc[0]); val y = ty(loc[1])
            val nodeIdx = index + 1
            val agentIdx = nodeVisitedBy[nodeIdx]
            if (agentIdx != null) {
                val color = agentColors[agentIdx % agentColors.size]
                // Filled ring: outer agent color, inner white core
                nodeFillPaint.color = color
                canvas.drawCircle(x, y, 13f, nodeFillPaint)
                nodeFillPaint.color = Color.WHITE
                canvas.drawCircle(x, y, 7f, nodeFillPaint)
                nodeOutlinePaint.color = color
                canvas.drawCircle(x, y, 13f, nodeOutlinePaint)
            } else {
                canvas.drawCircle(x, y, 9f, nodeUnvisitedPaint)
            }
        }

        // ── Depot (drawn last, always on top) ────────────────────────────────
        canvas.drawCircle(depotX, depotY, 22f, depotFillPaint)
        canvas.drawCircle(depotX, depotY, 22f, depotStrokePaint)
        canvas.drawText("S", depotX, depotY + 9f, depotLabelPaint)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun coordsFor(data: DatasetType.Synthetic, nodeIdx: Int): Pair<Float, Float> {
        val coords = when {
            nodeIdx == 0                  -> data.depot
            nodeIdx == data.loc.size + 1  -> data.depot
            else                          -> data.loc[nodeIdx - 1]
        }
        return tx(coords[0]) to ty(coords[1])
    }

    /**
     * Draw a directional arrowhead at the midpoint of the edge,
     * but only when the edge is long enough to avoid visual clutter.
     */
    private fun drawArrowIfLongEnough(
        canvas: Canvas,
        fx: Float, fy: Float, tx: Float, ty: Float,
        paint: Paint
    ) {
        val len = hypot(tx - fx, ty - fy)
        if (len < 40f) return

        val angle  = atan2(ty - fy, tx - fx)
        val mx     = (fx + tx) / 2
        val my     = (fy + ty) / 2
        val sz     = 14f

        val arrowPath = Path().apply {
            moveTo(mx, my)
            lineTo(
                mx - sz * cos(angle - Math.PI.toFloat() / 6),
                my - sz * sin(angle - Math.PI.toFloat() / 6)
            )
            moveTo(mx, my)
            lineTo(
                mx - sz * cos(angle + Math.PI.toFloat() / 6),
                my - sz * sin(angle + Math.PI.toFloat() / 6)
            )
        }
        canvas.drawPath(arrowPath, paint)
    }

    private fun drawGrid(canvas: Canvas) {
        val step = 80f
        var x = 0f
        while (x <= width) { canvas.drawLine(x, 0f, x, height.toFloat(), gridPaint); x += step }
        var y = 0f
        while (y <= height) { canvas.drawLine(0f, y, width.toFloat(), y, gridPaint); y += step }
    }

    private fun tx(x: Float) = width  * (padding + x * (1 - 2 * padding))
    private fun ty(y: Float) = height * (padding + y * (1 - 2 * padding))
}
