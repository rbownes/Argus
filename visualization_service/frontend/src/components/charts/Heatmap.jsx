import React from 'react'
import Plot from 'react-plotly.js'

const Heatmap = ({ title, data, options = {} }) => {
  const { x = [], y = [], z = [] } = data || {}

  // Default layout options
  const defaultLayout = {
    title: title || 'Theme Analysis Heatmap',
    width: options.width || undefined, // Responsive width
    height: options.height || 400,
    margin: {
      l: 50,
      r: 50,
      b: 100,
      t: 50,
      pad: 4
    },
    xaxis: {
      title: options.xAxisTitle || 'Themes',
      tickangle: -45
    },
    yaxis: {
      title: options.yAxisTitle || 'Models'
    }
  }

  // Default config options (for toolbar, etc)
  const defaultConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: [
      'lasso2d',
      'select2d',
      'autoScale2d',
      'resetScale2d',
      'hoverClosestCartesian',
      'hoverCompareCartesian'
    ]
  }

  // Merge with user options
  const layout = {
    ...defaultLayout,
    ...(options.layout || {})
  }

  const config = {
    ...defaultConfig,
    ...(options.config || {})
  }

  // Create colorscale based on score range (1-10)
  // This creates a gradient from red (low scores) to green (high scores)
  const colorscale = [
    [0, '#ef4444'],      // 0.0 -> red (low score)
    [0.3, '#f97316'],    // 0.3 -> orange
    [0.5, '#facc15'],    // 0.5 -> yellow
    [0.7, '#84cc16'],    // 0.7 -> light green
    [1.0, '#10b981']     // 1.0 -> green (high score)
  ]

  // Prepare the plot data
  const plotData = [
    {
      x,
      y,
      z,
      type: 'heatmap',
      colorscale: options.colorscale || colorscale,
      showscale: true,
      colorbar: {
        title: 'Score',
        titleside: 'right',
        tickvals: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        tickmode: 'array',
        ticktext: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        thickness: 20
      },
      zmin: 1,
      zmax: 10,
      hoverongaps: false,
      hovertemplate: 
        '<b>Model</b>: %{y}<br>' +
        '<b>Theme</b>: %{x}<br>' +
        '<b>Score</b>: %{z:.1f}<br>' +
        '<extra></extra>',
    }
  ]

  return (
    <div className="chart-container">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}

export default Heatmap
