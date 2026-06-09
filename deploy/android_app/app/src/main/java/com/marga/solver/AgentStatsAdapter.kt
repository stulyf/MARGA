package com.marga.solver

import android.graphics.drawable.GradientDrawable
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.marga.solver.databinding.ItemAgentStatBinding

/**
 * @param routes    per-agent node visit sequences
 * @param prizes    prize[i] = prize value of node i+1 (1-indexed), length = N
 * @param totalNodes  total number of delivery nodes (N), used for progress bar scale
 */
class AgentStatsAdapter(
    private val routes: List<List<Int>>,
    private val prizes: FloatArray = FloatArray(0),
    private val totalNodes: Int = 100,
) : RecyclerView.Adapter<AgentStatsAdapter.ViewHolder>() {

    private val colorResIds = listOf(
        R.color.agent_0, R.color.agent_1, R.color.agent_2,
        R.color.agent_3, R.color.agent_4
    )

    class ViewHolder(val binding: ItemAgentStatBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemAgentStatBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val route = routes[position]
        val b = holder.binding
        val color = ContextCompat.getColor(holder.itemView.context, colorResIds[position % colorResIds.size])

        // Color dot (oval drawable tinted)
        (b.viewColor.background as? GradientDrawable)?.setColor(color)
            ?: b.viewColor.setBackgroundColor(color)

        // Node count (exclude depot returns — nodeIdx 0 and N+1)
        val deliveryNodes = route.count { it in 1..totalNodes }
        b.tvAgentName.text = "Agent $position"
        b.tvNodesCount.text = "$deliveryNodes nodes"

        // Prize earned
        val prizeEarned = route.filter { it in 1..prizes.size }
            .sumOf { prizes[it - 1].toDouble() }.toFloat()
        b.tvPrize.text = "(+${"%.0f".format(prizeEarned)})"

        // Progress bar: fraction of total nodes visited by this agent
        val progressPct = if (totalNodes > 0) (deliveryNodes * 100) / totalNodes else 0
        b.progressPrize.setIndicatorColor(color)
        b.progressPrize.progress = progressPct
    }

    override fun getItemCount() = routes.size
}
