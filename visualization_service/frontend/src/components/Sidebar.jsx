import React from 'react'
import { NavLink } from 'react-router-dom'

const Sidebar = () => {
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
            <select className="form-control text-sm">
              <option>All Models</option>
              <option>GPT-4</option>
              <option>Claude-3</option>
              <option>Gemini Pro</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Themes</label>
            <select className="form-control text-sm">
              <option>All Themes</option>
              <option>science_explanations</option>
              <option>coding_tasks</option>
              <option>creative_writing</option>
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
