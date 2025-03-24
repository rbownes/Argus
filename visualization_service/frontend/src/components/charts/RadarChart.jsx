import React from 'react'
import { Radar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)

const RadarChart = ({ title, data, options = {} }) => {
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
      r: {
        min: 0,
        max: 10,
        beginAtZero: true,
        ticks: {
          stepSize: 2,
          display: true,
        },
        angleLines: {
          display: true,
        },
        pointLabels: {
          font: {
            size: 12,
          },
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
        backgroundColor: `${colors[index % colors.length]}33`, // Add transparency
        borderColor: colors[index % colors.length],
        borderWidth: 2,
        pointBackgroundColor: colors[index % colors.length],
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: colors[index % colors.length],
        pointRadius: 4,
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
      <Radar data={chartData} options={mergedOptions} />
    </div>
  )
}

export default RadarChart
