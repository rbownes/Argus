import React from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
)

const LineChart = ({ title, data, options = {} }) => {
  // Default chart options with sensible defaults
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
          text: 'Date',
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  }

  // Generate colors for series if not provided
  const chartData = {
    labels: data.labels || [],
    datasets: (data.series || []).map((series, index) => {
      // Default colors for up to 10 series
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
        label: series.name || `Series ${index + 1}`,
        data: series.data || [],
        borderColor: colors[index % colors.length],
        backgroundColor: `${colors[index % colors.length]}33`, // Add transparency
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 5,
        fill: false,
        ...series,
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
      <Line data={chartData} options={mergedOptions} />
    </div>
  )
}

export default LineChart
