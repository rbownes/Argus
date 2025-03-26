import React, { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import { getFilterOptions } from '../services/api'

const Sidebar = () => {
  const [filterOptions, setFilterOptions] = useState({
    models: [],
    themes: []
  })
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchFilterOptions() {
      try {
        setIsLoading(true)
        const options = await getFilterOptions()
        setFilterOptions(options)
        setError(null)
      } catch (err) {
        console.error('Error fetching filter options:', err)
        setError('Failed to load filters')
      } finally {
        setIsLoading(false)
      }
    }

    fetchFilterOptions()
  }, [])
  const activeClass = "bg-blue-100 text-blue-600 border-r-4 border-blue-600"
  const inactiveClass = "text-gray-700 hover:bg-gray-100"
  
  const navItems = [
    { to: "/", label: "Dashboard", icon: "📊" },
    { to: "/models", label: "Model Comparison", icon: "🤖" },
    { to: "/themes", label: "Theme Analysis", icon: "🔍" },
    { to: "/results", label: "Detailed Results", icon: "📋" },
  ]
  
  return (
    <aside className="sidebar w-64 overflow-y-auto border-r border-gray-200">
      <div className="py-2">
        <h3 className="text-xs uppercase font-semibold text-gray-500 px-4 mb-2">Navigation</h3>
        <nav className="mt-2">
          <ul>
            {navItems.map((item) => (
              <li key={item.to} className="mb-1">
                <NavLink
                  to={item.to}
                  className={({ isActive }) => 
                    `flex items-center px-4 py-2 text-sm font-medium ${isActive ? activeClass : inactiveClass}`
                  }
                >
                  <span className="mr-3">{item.icon}</span>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
      
      <div className="py-4 px-4 mt-4 border-t border-gray-200">
        <h3 className="text-xs uppercase font-semibold text-gray-500 mb-2">Filters</h3>
        <div className="text-sm text-gray-600">
          <div className="form-group">
            <label className="form-label">Date Range</label>
            <select className="form-control text-sm">
              <option>Last 7 days</option>
              <option>Last 30 days</option>
              <option>Last 90 days</option>
              <option>Custom...</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Models</label>
            <select className="form-control text-sm" disabled={isLoading}>
              <option>All Models</option>
              {isLoading ? (
                <option>Loading models...</option>
              ) : error ? (
                <option>Error loading models</option>
              ) : (
                filterOptions.models.map(model => (
                  <option key={model}>{model}</option>
                ))
              )}
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Themes</label>
            <select className="form-control text-sm" disabled={isLoading}>
              <option>All Themes</option>
              {isLoading ? (
                <option>Loading themes...</option>
              ) : error ? (
                <option>Error loading themes</option>
              ) : (
                filterOptions.themes.map(theme => (
                  <option key={theme}>{theme}</option>
                ))
              )}
            </select>
          </div>
          
          <button className="btn btn-primary w-full text-sm">
            Apply Filters
          </button>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
