import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { getDashboardSummary, getPerformanceTimeline } from '../services/api'
import LineChart from '../components/charts/LineChart'
import { format } from 'date-fns'

const Dashboard = () => {
  // State for date range filters
  const [dateRange, setDateRange] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    endDate: new Date(),
  })

  // Query summary data
  const summaryQuery = useQuery(
    ['dashboard-summary', dateRange],
    () => getDashboardSummary({
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
    }),
    {
      refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    }
  )

  // Query timeline data
  const timelineQuery = useQuery(
    ['performance-timeline', dateRange],
    () => getPerformanceTimeline({
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
      timeGrouping: 'day',
    }),
    {
      refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    }
  )

  const isLoading = summaryQuery.isLoading || timelineQuery.isLoading
  const isError = summaryQuery.isError || timelineQuery.isError
  const error = summaryQuery.error || timelineQuery.error

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="bg-red-50 p-4 rounded-lg border border-red-200 text-red-700">
        <h3 className="text-lg font-semibold mb-2">Error Loading Dashboard</h3>
        <p>{error?.message || 'An unexpected error occurred'}</p>
        <button 
          onClick={() => {
            summaryQuery.refetch()
            timelineQuery.refetch()
          }}
          className="mt-3 btn btn-danger"
        >
          Retry
        </button>
      </div>
    )
  }

  const summary = summaryQuery.data?.summary || {}
  const topModels = summaryQuery.data?.top_models || []
  const topThemes = summaryQuery.data?.top_themes || []
  const timeline = timelineQuery.data || {}

  // Format trend indicators
  const trendPercentage = summary.trend_percentage || 0
  const trendDirection = trendPercentage >= 0 ? 'up' : 'down'
  const trendColor = trendDirection === 'up' ? 'text-green-500' : 'text-red-500'
  const trendIcon = trendDirection === 'up' ? '↑' : '↓'

  return (
    <div className="container mx-auto">
      <h1 className="text-2xl font-bold mb-6">Model Evaluation Dashboard</h1>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="card">
          <h3 className="text-gray-500 text-sm uppercase">Total Evaluations</h3>
          <div className="text-3xl font-bold mt-2">{summary.total_evaluations || 0}</div>
        </div>
        
        <div className="card">
          <h3 className="text-gray-500 text-sm uppercase">Average Score</h3>
          <div className="text-3xl font-bold mt-2">{summary.avg_score?.toFixed(2) || 0}</div>
          {trendPercentage !== 0 && (
            <div className={`text-sm mt-1 ${trendColor}`}>
              {trendIcon} {Math.abs(trendPercentage).toFixed(1)}% from previous period
            </div>
          )}
        </div>
        
        <div className="card">
          <h3 className="text-gray-500 text-sm uppercase">Models Evaluated</h3>
          <div className="text-3xl font-bold mt-2">{summary.total_models || 0}</div>
        </div>
        
        <div className="card">
          <h3 className="text-gray-500 text-sm uppercase">Themes Covered</h3>
          <div className="text-3xl font-bold mt-2">{summary.total_themes || 0}</div>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Performance Over Time</h2>
        <div className="h-80">
          <LineChart
            data={timeline}
            options={{
              plugins: {
                title: {
                  display: false
                }
              }
            }}
          />
        </div>
      </div>

      {/* Top Performers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Top Performing Models</h2>
          <div className="overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. Score</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Evaluations</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topModels.map((model, index) => (
                  <tr key={index}>
                    <td className="px-4 py-2 text-sm font-medium text-gray-900">{model.model_id}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{model.avg_score.toFixed(2)}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{model.eval_count}</td>
                  </tr>
                ))}
                {topModels.length === 0 && (
                  <tr>
                    <td colSpan={3} className="px-4 py-2 text-sm text-gray-500 text-center">No data available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Top Performing Themes</h2>
          <div className="overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Theme</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. Score</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Evaluations</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topThemes.map((theme, index) => (
                  <tr key={index}>
                    <td className="px-4 py-2 text-sm font-medium text-gray-900">{theme.theme}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{theme.avg_score.toFixed(2)}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{theme.eval_count}</td>
                  </tr>
                ))}
                {topThemes.length === 0 && (
                  <tr>
                    <td colSpan={3} className="px-4 py-2 text-sm text-gray-500 text-center">No data available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      {/* Data Range Notice */}
      <div className="text-sm text-gray-500 text-center mb-4">
        Showing data from {format(dateRange.startDate, 'MMM d, yyyy')} to {format(dateRange.endDate, 'MMM d, yyyy')}
      </div>
    </div>
  )
}

export default Dashboard
