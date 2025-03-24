import React from 'react'
import { NavLink } from 'react-router-dom'

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-bold text-blue-600">Panopticon</h1>
        <span className="text-gray-500">Model Evaluation Dashboard</span>
      </div>
      
      <div className="flex items-center space-x-2">
        <a 
          href="/api/v1/dashboard/summary" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-sm text-gray-600 hover:text-blue-600"
        >
          API
        </a>
        <span className="text-gray-300">|</span>
        <a 
          href="https://github.com/yourusername/panopticon" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-sm text-gray-600 hover:text-blue-600"
        >
          GitHub
        </a>
      </div>
    </nav>
  )
}

export default Navbar
