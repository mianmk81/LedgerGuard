import { useState, useEffect } from 'react'
import { Bell, User, LogOut, RefreshCw } from 'lucide-react'
import { api } from '../../api/client'

export default function Header({ onLogout }) {
  const [connectionStatus, setConnectionStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchConnectionStatus()
  }, [])

  const fetchConnectionStatus = async () => {
    try {
      const status = await api.connection.getStatus()
      setConnectionStatus(status)
    } catch (error) {
      console.error('Failed to fetch connection status:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
      <div className="flex items-center space-x-4">
        {/* Connection Status */}
        {!loading && connectionStatus && (
          <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-sm font-medium text-green-700">
              {connectionStatus.company_name || 'Connected'}
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center space-x-4">
        {/* Refresh */}
        <button
          onClick={fetchConnectionStatus}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          aria-label="Refresh"
        >
          <RefreshCw className="w-5 h-5 text-gray-600" />
        </button>

        {/* Notifications */}
        <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative">
          <Bell className="w-5 h-5 text-gray-600" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        {/* User Menu */}
        <div className="relative group">
          <button className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors">
            <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-primary-700" />
            </div>
            <span className="text-sm font-medium text-gray-700">Admin</span>
          </button>

          {/* Dropdown Menu */}
          <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
            <button
              onClick={onLogout}
              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
