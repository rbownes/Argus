import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ModelComparison from './pages/ModelComparison'
import ThemeAnalysis from './pages/ThemeAnalysis'
import DetailedResults from './pages/DetailedResults'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'

function App() {
  return (
    <div className="flex flex-col h-screen">
      {/* Top Navigation */}
      <Navbar />
      
      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <main className="flex-1 overflow-auto p-4 bg-gray-100">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<ModelComparison />} />
            <Route path="/themes" element={<ThemeAnalysis />} />
            <Route path="/results" element={<DetailedResults />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
