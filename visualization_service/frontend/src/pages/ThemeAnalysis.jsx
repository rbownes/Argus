import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { getThemeAnalysis, getFilterOptions } from '../services/api'
import Heatmap from '../components/charts/Heatmap'
import LineChart from '../components/charts/LineChart'
import { format } from 'date-fns'

const ThemeAnalysis = () => {
  // State for filters
  const [filters, setFilters] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    endDate: new Date(),
    selectedModels: [],
    selectedThemes: [],
  })

  // Query theme analysis data
  const analysisQuery = useQuery(
    ['theme-analysis', filters],
    () => getThemeAnalysis({
      startDate: filters.startDate,
      endDate: filters.endDate,
      models: filters.selectedModels.length > 0 ? filters.selectedModels : undefined,
      themes: filters.selectedThemes.length > 0 ? filters.selectedThemes : undefined,
    }),
    {
      refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    }
  )

  // Query available filter options
  const filterOptionsQuery = useQuery(
    'filter-options',
    getFilterOptions,
    {
      staleTime: 5 * 60 * 1000, // Consider data fresh for 5 minutes
    }
  )

  const isLoading = analysisQuery.isLoading || filterOptionsQuery.isLoading
  const isError = analysisQuery.isError || filterOptionsQuery.isError
  const error = analysisQuery.error || filterOptionsQuery.error

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading theme analysis data...</p>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="bg-red-50 p-4 rounded-lg border border-red-200 text-red-700">
        <h3 className="text-lg font-semibold mb-2">Error Loading Data</h3>
        <p>{error?.message || 'An unexpected error occurred'}</p>
        <button 
          onClick={() => {
            analysisQuery.refetch()
            filterOptionsQuery.refetch()
          }}
          className="mt-3 btn btn-danger"
        >
          Retry
        </button>
      </div>
    )
  }

  // Extract data
  const analysisData = analysisQuery.data || {}
  const heatmapData = analysisData.heatmap || { x: [], y: [], z: [] }
  const themeTimelineData = analysisData.theme_timeline || { labels: [], datasets: [] }
  const rawData = analysisData.raw_data || []
  
  // Get available filter options
  const availableModels = filterOptionsQuery.data?.models || []
  const availableThemes = filterOptionsQuery.data?.themes || []

  // Handle filter changes
  const handleModelFilterChange = (e) => {
    const selectedOptions = Array.from(
      e.target.selectedOptions,
      (option) => option.value
    )
    setFilters((prev) => ({
      ...prev,
      selectedModels: selectedOptions,
    }))
  }

  const handleThemeFilterChange = (e) => {
    const selectedOptions = Array.from(
      e.target.selectedOptions,
      (option) => option.value
    )
    setFilters((prev) => ({
      ...prev,
      selectedThemes: selectedOptions,
    }))
  }

  const handleDateRangeChange = (e) => {
    const value = e.target.value
    const now = new Date()
    let startDate
    
    switch (value) {
      case '7days':
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
        break
      case '30days':
        startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
        break
      case '90days':
        startDate = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000)
        break
      default:
        startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
    }
    
    setFilters((prev) => ({
      ...prev,
      startDate,
      endDate: now,
    }))
  }

  return (
    <div className="container mx-auto">
      <h1 className="text-2xl font-bold mb-6">Theme Analysis</h1>
      
      {/* Filters */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Filters</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="form-group">
            <label className="form-label">Date Range</label>
            <select 
              className="form-control"
              onChange={handleDateRangeChange}
              defaultValue="30days"
            >
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
              <option value="90days">Last 90 Days</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Models</label>
            <select 
              className="form-control" 
              multiple 
              size={3}
              onChange={handleModelFilterChange}
              value={filters.selectedModels}
            >
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-500 mt-1">
              Hold Ctrl/Cmd to select multiple
            </div>
          </div>
          
          <div className="form-group">
            <label className="form-label">Themes</label>
            <select 
              className="form-control" 
              multiple 
              size={3}
              onChange={handleThemeFilterChange}
              value={filters.selectedThemes}
            >
              {availableThemes.map((theme) => (
                <option key={theme} value={theme}>
                  {theme}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-500 mt-1">
              Hold Ctrl/Cmd to select multiple
            </div>
          </div>
        </div>
      </div>
      
      {/* Heatmap */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Theme Performance Heatmap</h2>
        <div className="h-96">
          <Heatmap
            title="Model Performance by Theme"
            data={heatmapData}
            options={{
              height: 500,
              xAxisTitle: 'Themes',
              yAxisTitle: 'Models',
            }}
          />
        </div>
        <div className="text-sm text-gray-600 mt-2">
          Hover over cells to see detailed score information. Brighter green indicates higher scores.
        </div>
      </div>
      
      {/* Theme Performance Timeline */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Theme Performance Over Time</h2>
        <div className="h-80">
          <LineChart
            data={themeTimelineData}
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
      
      {/* Detailed Data Table */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Detailed Score Data</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Theme</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {rawData.map((row, index) => (
                <tr key={index}>
                  <td className="px-4 py-2 text-sm font-medium text-gray-900">{row.model_id}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.theme}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.score?.toFixed(2)}</td>
                </tr>
              ))}
              {rawData.length === 0 && (
                <tr>
                  <td colSpan={3} className="px-4 py-2 text-sm text-gray-500 text-center">No data available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Data Range Notice */}
      <div className="text-sm text-gray-500 text-center mb-4">
        Showing data from {format(filters.startDate, 'MMM d, yyyy')} to {format(filters.endDate, 'MMM d, yyyy')}
      </div>
    </div>
  )
}

export default ThemeAnalysis
