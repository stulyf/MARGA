package com.marga.solver

import android.app.Dialog
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment

class BenchmarkDialog : DialogFragment() {

    var avg: Long = 0
    var min: Long = 0
    var max: Long = 0

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        return AlertDialog.Builder(requireContext())
            .setTitle("Benchmark Results (10 runs)")
            .setMessage("Average:  $avg ms\nMinimum:  $min ms\nMaximum:  $max ms")
            .setPositiveButton("OK", null)
            .create()
    }
}
