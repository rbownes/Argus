import axios from 'axios'

// Create an axios instance with common configuration
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add a request interceptor to include API key
api.interceptors.request.use(
  (config) => {
    // Get API key from environment variable set during build
    const apiKey = import.meta.env.VITE_API_KEY;
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    } else {
      console.warn("API Key (VITE_API_KEY) is not set. Requests might fail.");
    }
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
  const { startDate, endDate, models, themes, providers } = filters
  
  const params = new URLSearchParams()
  if (startDate) params.append('start_date', startDate.toISOString())
  if (endDate) params.append('end_date', endDate.toISOString())
  if (models?.length) params.append('models', models.join(','))
  if (themes?.length) params.append('themes', themes.join(','))
  if (providers?.length) params.append('providers', providers.join(','))
  
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
  try {
    // First try to get models from judge-models endpoint
    const judgeModelsResponse = await api.get('/judge-models')
    const judgeModels = judgeModelsResponse.data?.models || []
    
    // Also get other filters from regular filters endpoint
    const filtersResponse = await api.get('/dashboard/filters')
    const filters = filtersResponse.data || {}
    
    // Extract provider information from models
    const providers = [...new Set(
      judgeModels
        .filter(model => model.provider)
        .map(model => model.provider)
    )]
    
    // Use model objects directly, ensuring we have a consistent format
    const modelObjects = judgeModels.map(model => {
      // If it's already an object with all required fields, use it
      if (typeof model === 'object' && model.id && model.provider) {
        return model
      }
      
      // If it's an object missing provider
      if (typeof model === 'object' && model.id) {
        return {
          ...model,
          provider: model.provider || 'unknown'
        }
      }
      
      // If it's just a string ID
      return {
        id: model,
        name: model,
        provider: 'unknown'
      }
    })
    
    // Get unique model IDs for backwards compatibility
    const modelIds = [...new Set(modelObjects.map(model => model.id))]
    
    // Include any models from filters endpoint not already in the list
    const additionalModelIds = (filters.models || []).filter(
      id => !modelIds.includes(id)
    )
    
    const additionalModels = additionalModelIds.map(id => ({
      id,
      name: id,
      provider: 'unknown'
    }))
    
    // Make sure there are default providers if none were found
    const defaultProviders = ['openai', 'anthropic', 'google']
    const allProviders = providers.length ? providers : defaultProviders
    
    return {
      ...filters,
      models: modelObjects.concat(additionalModels),
      modelIds: modelIds.concat(additionalModelIds),
      providers: allProviders
    }
  } catch (error) {
    console.error('Error getting filter options:', error)
    // Fallback to just the filters endpoint if something goes wrong
    try {
      const { data } = await api.get('/dashboard/filters')
      return {
        ...data,
        providers: ['openai', 'anthropic', 'google'],
        modelIds: data.models || []
      }
    } catch (fallbackError) {
      console.error('Fallback error:', fallbackError)
      return { models: [], themes: [], providers: ['openai', 'anthropic', 'google'], modelIds: [] }
    }
  }
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
