import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect } from 'react'

// Layout components
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'

// Page components
import ConnectionSetup from './pages/ConnectionSetup'
import HealthDashboard from './pages/HealthDashboard'
import IncidentsList from './pages/IncidentsList'
import IncidentDetail from './pages/IncidentDetail'
import PostmortemView from './pages/PostmortemView'
import MonitorsDashboard from './pages/MonitorsDashboard'
import ComparisonSimulation from './pages/ComparisonSimulation'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Check for stored auth token on mount
  useEffect(() => {
    const token = localStorage.getItem('ledgerguard_token')
    if (token) {
      setIsAuthenticated(true)
    }
  }, [])

  const handleLogin = (token, realmId) => {
    localStorage.setItem('ledgerguard_token', token)
    localStorage.setItem('ledgerguard_realm_id', realmId)
    setIsAuthenticated(true)
  }

  const handleLogout = () => {
    localStorage.removeItem('ledgerguard_token')
    localStorage.removeItem('ledgerguard_realm_id')
    setIsAuthenticated(false)
  }

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {isAuthenticated ? (
          <div className="flex h-screen">
            {/* Sidebar */}
            <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

            {/* Main content */}
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header onLogout={handleLogout} />

              <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<HealthDashboard />} />
                  <Route path="/incidents" element={<IncidentsList />} />
                  <Route path="/incidents/:incidentId" element={<IncidentDetail />} />
                  <Route path="/incidents/:incidentId/postmortem" element={<PostmortemView />} />
                  <Route path="/monitors" element={<MonitorsDashboard />} />
                  <Route path="/comparison" element={<ComparisonSimulation />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </main>
            </div>
          </div>
        ) : (
          <Routes>
            <Route path="/setup" element={<ConnectionSetup onLogin={handleLogin} />} />
            <Route path="*" element={<Navigate to="/setup" replace />} />
          </Routes>
        )}
      </div>
    </BrowserRouter>
  )
}

export default App
