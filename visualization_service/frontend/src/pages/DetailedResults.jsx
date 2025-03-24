import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { getDetailedResults, getFilterOptions } from '../services/api'
import { format } from 'date-fns'

const DetailedResults = () => {
  // State for filters and pagination
  const [filters, setFilters] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    endDate: new Date(),
    selectedModels: [],
    selectedThemes: [],
    selectedPrompts: [],
    minScore: undefined,
    maxScore: undefined,
    page: 1,
    pageSize: 10,
    sortBy: 'timestamp',
    sortDesc: true,
  })

  // Query detailed results data
  const resultsQuery = useQuery(
    ['detailed-results', filters],
    () => getDetailedResults({
      startDate: filters.startDate,
      endDate: filters.endDate,
      models: filters.selectedModels.length > 0 ? filters.selectedModels : undefined,
      themes: filters.selectedThemes.length > 0 ? filters.selectedThemes : undefined,
      evaluationPrompts: filters.selectedPrompts.length > 0 ? filters.selectedPrompts : undefined,
      minScore: filters.minScore,
      maxScore: filters.maxScore,
      page: filters.page,
      pageSize: filters.pageSize,
      sortBy: filters.sortBy,
      sortDesc: filters.sortDesc,
    }),
    {
      keepPreviousData: true, // Keep previous data while fetching new data
      refetchInterval: 30 * 1000, // Refetch every 30 seconds
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

  const isLoading = resultsQuery.isLoading || filterOptionsQuery.isLoading
  const isError = resultsQuery.isError || filterOptionsQuery.isError
  const error = resultsQuery.error || filterOptionsQuery.error

  if (isLoading && !resultsQuery.data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading detailed results...</p>
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
            resultsQuery.refetch()
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
  const results = resultsQuery.data?.results || []
  const pagination = resultsQuery.data?.pagination || { 
    page: 1, 
    pageSize: 10, 
    totalCount: 0, 
    totalPages: 0, 
    hasNext: false, 
    hasPrev: false 
  }
  
  // Get available filter options
  const availableModels = filterOptionsQuery.data?.models || []
  const availableThemes = filterOptionsQuery.data?.themes || []
  const availablePrompts = filterOptionsQuery.data?.evaluation_prompts || []

  // Handle filter changes
  const handleModelFilterChange = (e) => {
    const selectedOptions = Array.from(
      e.target.selectedOptions,
      (option) => option.value
    )
    setFilters((prev) => ({
      ...prev,
      selectedModels: selectedOptions,
      page: 1, // Reset to first page when filter changes
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
      page: 1, // Reset to first page when filter changes
    }))
  }

  const handlePromptFilterChange = (e) => {
    const selectedOptions = Array.from(
      e.target.selectedOptions,
      (option) => option.value
    )
    setFilters((prev) => ({
      ...prev,
      selectedPrompts: selectedOptions,
      page: 1, // Reset to first page when filter changes
    }))
  }

  const handleScoreFilterChange = (e, type) => {
    const value = e.target.value ? parseFloat(e.target.value) : undefined
    setFilters((prev) => ({
      ...prev,
      [type]: value,
      page: 1, // Reset to first page when filter changes
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
      page: 1, // Reset to first page when filter changes
    }))
  }

  const handleSortChange = (sortField) => {
    setFilters((prev) => ({
      ...prev,
      sortBy: sortField,
      sortDesc: prev.sortBy === sortField ? !prev.sortDesc : true,
    }))
  }

  // Pagination handlers
  const goToPage = (page) => {
    setFilters((prev) => ({
      ...prev,
      page,
    }))
  }

  const handlePageSizeChange = (e) => {
    const pageSize = parseInt(e.target.value, 10)
    setFilters((prev) => ({
      ...prev,
      pageSize,
      page: 1, // Reset to first page when page size changes
    }))
  }

  // Generate page numbers for pagination
  const getPageNumbers = () => {
    const { page, totalPages } = pagination
    const pageNumbers = []
    const maxPagesToShow = 5
    
    if (totalPages <= maxPagesToShow) {
      // Show all pages if total pages is less than or equal to max pages to show
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i)
      }
    } else {
      // Show a subset of pages
      const halfMaxPages = Math.floor(maxPagesToShow / 2)
      
      // Always include first page
      pageNumbers.push(1)
      
      // Calculate start and end page numbers
      let startPage = Math.max(2, page - halfMaxPages)
      let endPage = Math.min(totalPages - 1, page + halfMaxPages)
      
      // Adjust if we're near the start or end
      if (startPage === 2) {
        endPage = Math.min(totalPages - 1, startPage + maxPagesToShow - 3)
      }
      if (endPage === totalPages - 1) {
        startPage = Math.max(2, endPage - (maxPagesToShow - 3))
      }
      
      // Add ellipsis if needed
      if (startPage > 2) {
        pageNumbers.push('...')
      }
      
      // Add middle pages
      for (let i = startPage; i <= endPage; i++) {
        pageNumbers.push(i)
      }
      
      // Add ellipsis if needed
      if (endPage < totalPages - 1) {
        pageNumbers.push('...')
      }
      
      // Always include last page
      pageNumbers.push(totalPages)
    }
    
    return pageNumbers
  }

  // Format result data for display
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM d, yyyy HH:mm:ss')
    } catch (e) {
      return dateString
    }
  }

  return (
    <div className="container mx-auto">
      <h1 className="text-2xl font-bold mb-6">Detailed Evaluation Results</h1>
      
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
          
          <div className="form-group">
            <label className="form-label">Evaluation Prompts</label>
            <select 
              className="form-control" 
              multiple 
              size={3}
              onChange={handlePromptFilterChange}
              value={filters.selectedPrompts}
            >
              {availablePrompts.map((prompt) => (
                <option key={prompt.id} value={prompt.id}>
                  {prompt.id}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-500 mt-1">
              Hold Ctrl/Cmd to select multiple
            </div>
          </div>
          
          <div className="form-group">
            <label className="form-label">Minimum Score</label>
            <input 
              type="number" 
              className="form-control" 
              min="1" 
              max="10" 
              step="0.1"
              value={filters.minScore || ''}
              onChange={(e) => handleScoreFilterChange(e, 'minScore')}
              placeholder="Minimum score (1-10)"
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">Maximum Score</label>
            <input 
              type="number" 
              className="form-control" 
              min="1" 
              max="10" 
              step="0.1"
              value={filters.maxScore || ''}
              onChange={(e) => handleScoreFilterChange(e, 'maxScore')}
              placeholder="Maximum score (1-10)"
            />
          </div>
        </div>
      </div>
      
      {/* Results count and pagination controls */}
      <div className="flex flex-wrap justify-between items-center mb-4">
        <div className="text-sm text-gray-600">
          Showing {results.length} of {pagination.totalCount} results
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Rows per page:</span>
          <select 
            className="form-control text-sm py-1 px-2"
            value={filters.pageSize}
            onChange={handlePageSizeChange}
          >
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>
      
      {/* Results Table */}
      <div className="card mb-6 overflow-hidden">
        {resultsQuery.isFetching && (
          <div className="absolute inset-0 bg-white bg-opacity-70 flex items-center justify-center z-10">
            <div className="spinner w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th 
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortChange('timestamp')}
                >
                  <div className="flex items-center">
                    Date
                    {filters.sortBy === 'timestamp' && (
                      <span className="ml-1">{filters.sortDesc ? '↓' : '↑'}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortChange('model_id')}
                >
                  <div className="flex items-center">
                    Model
                    {filters.sortBy === 'model_id' && (
                      <span className="ml-1">{filters.sortDesc ? '↓' : '↑'}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortChange('theme')}
                >
                  <div className="flex items-center">
                    Theme
                    {filters.sortBy === 'theme' && (
                      <span className="ml-1">{filters.sortDesc ? '↓' : '↑'}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortChange('score')}
                >
                  <div className="flex items-center">
                    Score
                    {filters.sortBy === 'score' && (
                      <span className="ml-1">{filters.sortDesc ? '↓' : '↑'}</span>
                    )}
                  </div>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Query
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {results.map((result) => (
                <tr key={result.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-500 whitespace-nowrap">
                    {formatDate(result.timestamp)}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {result.model_id}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {result.theme}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span 
                      className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        result.score >= 8 ? 'bg-green-100 text-green-800' : 
                        result.score >= 6 ? 'bg-blue-100 text-blue-800' :
                        result.score >= 4 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}
                    >
                      {result.score.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    <div className="max-w-xs truncate" title={result.query_text}>
                      {result.query_text}
                    </div>
                  </td>
                </tr>
              ))}
              {results.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-3 text-sm text-gray-500 text-center">
                    No results found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Pagination */}
      {pagination.totalPages > 1 && (
        <div className="flex justify-center mb-6">
          <nav className="flex items-center">
            <button
              onClick={() => goToPage(pagination.page - 1)}
              disabled={!pagination.hasPrev}
              className={`px-3 py-1 rounded-l-md border ${
                pagination.hasPrev 
                  ? 'bg-white hover:bg-gray-50 text-blue-600 border-gray-300' 
                  : 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
              }`}
            >
              Previous
            </button>
            
            {getPageNumbers().map((pageNum, index) => (
              <button
                key={index}
                onClick={() => typeof pageNum === 'number' ? goToPage(pageNum) : null}
                disabled={pageNum === '...'}
                className={`px-3 py-1 border-t border-b ${
                  pageNum === pagination.page
                    ? 'bg-blue-50 text-blue-600 font-medium border-blue-500'
                    : pageNum === '...'
                    ? 'bg-gray-50 text-gray-400 border-gray-200'
                    : 'bg-white hover:bg-gray-50 text-blue-600 border-gray-300'
                } ${index === 0 ? 'border-l' : ''}`}
              >
                {pageNum}
              </button>
            ))}
            
            <button
              onClick={() => goToPage(pagination.page + 1)}
              disabled={!pagination.hasNext}
              className={`px-3 py-1 rounded-r-md border ${
                pagination.hasNext 
                  ? 'bg-white hover:bg-gray-50 text-blue-600 border-gray-300' 
                  : 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
              }`}
            >
              Next
            </button>
          </nav>
        </div>
      )}
      
      {/* Data Range Notice */}
      <div className="text-sm text-gray-500 text-center mb-4">
        Showing data from {format(filters.startDate, 'MMM d, yyyy')} to {format(filters.endDate, 'MMM d, yyyy')}
      </div>
    </div>
  )
}

export default DetailedResults
