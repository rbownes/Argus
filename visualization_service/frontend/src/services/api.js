import axios from 'axios'

// Create an axios instance with common configuration
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add a request interceptor to include API key
// In a production app, this would come from a secure source like an env variable or auth provider
api.interceptors.request.use(
  (config) => {
    // Add API key to every request
    // For demo purposes, we'll use a hardcoded key that matches the backend default
    // In production, you'd get this from a more secure source
    config.headers['X-API-Key'] = 'development_key'
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// ---- Dashboard API ----

export const getDashboardSummary = async (filters = {}) => {
  const { startDate, endDate, models, themes } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  
  const { data } = await api.get(`/dashboard/summary?${params.toString()}`)
  return data
}

export const getPerformanceTimeline = async (filters = {}) => {
  const { startDate, endDate, models, themes, timeGrouping = 'day' } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  params.append('time_grouping', timeGrouping)
  
  const { data } = await api.get(`/dashboard/timeline?${params.toString()}`)
  return data
}

export const getModelComparison = async (filters = {}) => {
  const { startDate, endDate, models, themes } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  
  const { data } = await api.get(`/dashboard/models?${params.toString()}`)
  return data
}

export const getThemeAnalysis = async (filters = {}) => {
  const { startDate, endDate, models, themes } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  
  const { data } = await api.get(`/dashboard/themes?${params.toString()}`)
  return data
}

export const getDetailedResults = async (filters = {}) => {
  const { 
    startDate, 
    endDate, 
    models, 
    themes, 
    evaluationPrompts,
    minScore,
    maxScore,
    page = 1,
    pageSize = 10,
    sortBy = 'timestamp',
    sortDesc = true
  } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  if (evaluationPrompts?.length) params.append('evaluation_prompts', evaluationPrompts.join(','))
  if (minScore !== undefined) params.append('min_score', minScore)
  if (maxScore !== undefined) params.append('max_score', maxScore)
  params.append('page', page)
  params.append('page_size', pageSize)
  params.append('sort_by', sortBy)
  params.append('sort_desc', sortDesc)
  
  const { data } = await api.get(`/dashboard/results?${params.toString()}`)
  return data
}

export const getFilterOptions = async () => {
  const { data } = await api.get('/dashboard/filters')
  return data
}

// WebSocket connection for real-time updates (optional)
export const createWebSocketConnection = () => {
  // Determine WebSocket URL (ws or wss depending on http/https)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws`
  
  const socket = new WebSocket(wsUrl)
  
  socket.onopen = () => {
    console.log('WebSocket connection established')
  }
  
  socket.onclose = () => {
    console.log('WebSocket connection closed')
  }
  
  socket.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
  
  return socket
}
