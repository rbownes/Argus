import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { getModelComparison, getFilterOptions } from '../services/api'
import BarChart from '../components/charts/BarChart'
import RadarChart from '../components/charts/RadarChart'
import { format } from 'date-fns'

const ModelComparison = () => {
  // State for filters
  const [filters, setFilters] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    endDate: new Date(),
    selectedModels: [],
    selectedThemes: [],
    selectedProviders: [],
  })

  // Query model comparison data
  const comparisonQuery = useQuery(
    ['model-comparison', filters],
    () => getModelComparison({
      startDate: filters.startDate,
      endDate: filters.endDate,
      models: filters.selectedModels.length > 0 ? filters.selectedModels : undefined,
      themes: filters.selectedThemes.length > 0 ? filters.selectedThemes : undefined,
      providers: filters.selectedProviders.length > 0 ? filters.selectedProviders : undefined,
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

  const isLoading = comparisonQuery.isLoading || filterOptionsQuery.isLoading
  const isError = comparisonQuery.isError || filterOptionsQuery.isError
  const error = comparisonQuery.error || filterOptionsQuery.error

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading comparison data...</p>
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
            comparisonQuery.refetch()
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
  const comparisonData = comparisonQuery.data || {}
  const barChartData = comparisonData.bar_chart || { labels: [], datasets: [] }
  const radarChartData = comparisonData.radar_chart || { labels: [], datasets: [] }
  const histogramData = comparisonData.histograms || {}
  const rawData = comparisonData.raw_data || []
  
  // Get available model IDs for dropdown
  const availableModels = filterOptionsQuery.data?.modelIds || []
  
  // Get available themes
  const availableThemes = filterOptionsQuery.data?.themes || []
  
  // Get providers directly from the API response
  const availableProviders = filterOptionsQuery.data?.providers || ['openai', 'anthropic', 'google']

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
  
  const handleProviderFilterChange = (e) => {
    const selectedOptions = Array.from(
      e.target.selectedOptions,
      (option) => option.value
    )
    setFilters((prev) => ({
      ...prev,
      selectedProviders: selectedOptions,
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
      <h1 className="text-2xl font-bold mb-6">Model Comparison</h1>
      
      {/* Filters */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Filters</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
            <label className="form-label">Providers</label>
            <select 
              className="form-control" 
              multiple 
              size={3}
              onChange={handleProviderFilterChange}
              value={filters.selectedProviders}
            >
              {availableProviders.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
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
      
      {/* Performance Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Bar Chart */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Average Performance by Model</h2>
          <div className="h-80">
            <BarChart
              data={barChartData}
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
        
        {/* Radar Chart */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Performance Across Themes</h2>
          <div className="h-80">
            <RadarChart
              data={radarChartData}
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
      </div>
      
      {/* Detailed Data Table */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">Detailed Comparison Data</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Provider</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Theme</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. Score</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Min Score</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max Score</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {rawData.map((row, index) => (
                <tr key={index}>
                  <td className="px-4 py-2 text-sm font-medium text-gray-900">{row.model_id}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.provider || 'unknown'}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.theme}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.avg_score?.toFixed(2)}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.min_score?.toFixed(2)}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.max_score?.toFixed(2)}</td>
                  <td className="px-4 py-2 text-sm text-gray-500">{row.eval_count}</td>
                </tr>
              ))}
              {rawData.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-2 text-sm text-gray-500 text-center">No data available</td>
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

export default ModelComparison
