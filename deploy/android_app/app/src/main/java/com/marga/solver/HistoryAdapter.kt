package com.marga.solver

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.marga.solver.databinding.ItemHistoryBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryAdapter : ListAdapter<SolverHistory, HistoryAdapter.ViewHolder>(DIFF) {

    class ViewHolder(val binding: ItemHistoryBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemHistoryBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = getItem(position)
        val b = holder.binding

        b.tvRewardBadge.text = item.result.totalPrize.toInt().toString()
        b.tvDatasetType.text = item.datasetType
        b.tvNodeCount.text = "${item.nodeCount} nodes · ${item.result.routes.size} agents · ${item.datasetType}"
        b.chipLatency.text = "${item.result.inferenceTimeMs} ms"

        val elapsed = System.currentTimeMillis() - item.timestamp
        b.tvTimestamp.text = when {
            elapsed < 60_000    -> "Just now"
            elapsed < 3_600_000 -> "${elapsed / 60_000} min ago"
            else                -> SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(item.timestamp))
        }
    }

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<SolverHistory>() {
            override fun areItemsTheSame(a: SolverHistory, b: SolverHistory) =
                a.timestamp == b.timestamp
            override fun areContentsTheSame(a: SolverHistory, b: SolverHistory) = a == b
        }
    }
}
