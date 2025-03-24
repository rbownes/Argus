import React from 'react'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

const BarChart = ({ title, data, options = {} }) => {
  // Default chart options
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        align: 'end',
      },
      title: {
        display: !!title,
        text: title || '',
        font: {
          size: 16,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      y: {
        min: 0,
        max: 10,
        beginAtZero: true,
        title: {
          display: true,
          text: 'Score',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Models',
        },
      },
    },
  }

  // Generate colors for datasets if not provided
  const chartData = {
    labels: data.labels || [],
    datasets: (data.datasets || []).map((dataset, index) => {
      // Default colors for up to 10 datasets
      const colors = [
        '#3b82f6', // blue
        '#10b981', // green
        '#8b5cf6', // purple
        '#ef4444', // red
        '#f59e0b', // amber
        '#6366f1', // indigo
        '#ec4899', // pink
        '#14b8a6', // teal
        '#f97316', // orange
        '#6b7280', // gray
      ]

      return {
        label: dataset.label || `Dataset ${index + 1}`,
        data: dataset.data || [],
        backgroundColor: colors[index % colors.length],
        borderColor: colors[index % colors.length],
        borderWidth: 1,
        ...dataset,
      }
    }),
  }

  // Merge default options with provided options
  const mergedOptions = {
    ...defaultOptions,
    ...options,
    plugins: {
      ...defaultOptions.plugins,
      ...(options.plugins || {}),
    },
    scales: {
      ...defaultOptions.scales,
      ...(options.scales || {}),
    },
  }

  return (
    <div className="chart-container">
      <Bar data={chartData} options={mergedOptions} />
    </div>
  )
}

export default BarChart
