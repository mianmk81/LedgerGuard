import {
    ArrowTrendingUpIcon,
    BeakerIcon,
    ChartBarIcon,
    ChartPieIcon,
    ChevronDownIcon,
    ChevronLeftIcon,
    ChevronRightIcon,
    ChevronUpIcon,
    CogIcon,
    CpuChipIcon,
    ExclamationTriangleIcon,
    HeartIcon,
    HomeIcon,
    LightBulbIcon,
} from '@heroicons/react/24/outline'
import { useEffect, useState, useCallback } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { api } from '../../api/client'

const primaryNav = [
  { name: 'Setup', href: '/setup', icon: CogIcon },
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Forecast', href: '/dashboard#future-score', icon: ArrowTrendingUpIcon },
  { name: 'Models', href: '/models', icon: CpuChipIcon },
  { name: 'Credit Pulse', href: '/credit-pulse', icon: HeartIcon },
  { name: 'Insights', href: '/insights', icon: LightBulbIcon },
  { name: 'Incidents', href: '/incidents', icon: ExclamationTriangleIcon },
]

const moreNav = [
  { name: 'Monitors', href: '/monitors', icon: ChartBarIcon },
  { name: 'Analysis', href: '/analysis', icon: BeakerIcon },
]

export default function Sidebar({ isOpen, onToggle, onNavigate }) {
  const location = useLocation()
  const [connectionStatus, setConnectionStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showMoreNav, setShowMoreNav] = useState(false)

  const fetchConnectionStatus = useCallback(async () => {
    try {
      const status = await api.connection.getStatus()
      setConnectionStatus(status)
    } catch (error) {
      console.error('Failed to fetch connection status:', error)
      setConnectionStatus(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchConnectionStatus()
    const interval = setInterval(fetchConnectionStatus, 30000) // Poll every 30s
    return () => clearInterval(interval)
  }, [fetchConnectionStatus])

  const getStatusColor = () => {
    if (!connectionStatus) return 'bg-gray-400'
    if (connectionStatus.is_valid && connectionStatus.is_active) return 'bg-green-500'
    if (connectionStatus.is_valid && !connectionStatus.is_active) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getStatusText = () => {
    if (loading) return 'Checking...'
    if (!connectionStatus) return 'Disconnected'
    if (connectionStatus.is_valid && connectionStatus.is_active) return 'Connected'
    if (connectionStatus.is_valid && !connectionStatus.is_active) return 'Inactive'
    return 'Error'
  }

  return (
    <div
      className={`${
        isOpen ? 'w-64' : 'w-20'
      } bg-white border-r border-slate-200/80 transition-all duration-300 ease-in-out flex flex-col shadow-sm`}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-slate-200/80">
        {isOpen && (
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 bg-gradient-to-br from-primary-600 to-primary-700 rounded-xl flex items-center justify-center shadow-md ring-1 ring-primary-500/20">
              <ChartPieIcon className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl text-slate-800 tracking-tight">LedgerGuard</span>
          </div>
        )}
        {!isOpen && (
          <div className="w-9 h-9 bg-gradient-to-br from-primary-600 to-primary-700 rounded-xl flex items-center justify-center shadow-md mx-auto">
            <ChartPieIcon className="w-5 h-5 text-white" />
          </div>
        )}
        <button
          onClick={onToggle}
          className="p-2 rounded-xl hover:bg-slate-100 active:bg-slate-200 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          aria-label={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          {isOpen ? (
            <ChevronLeftIcon className="w-5 h-5 text-slate-500" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-slate-500" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav aria-label="Main navigation" className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {primaryNav.map((item) => {
          const fullPath = location.pathname + (location.hash || '')
          const isActive = item.href.includes('#')
            ? fullPath === item.href
            : (location.pathname === item.href || location.pathname.startsWith(item.href + '/'))
          const Icon = item.icon

          return (
            <Link
              key={item.name}
              to={item.href}
              className={`nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
              title={!isOpen ? item.name : undefined}
              aria-current={isActive ? 'page' : undefined}
              onClick={onNavigate}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-primary-600' : 'text-slate-400'}`} />
              {isOpen && <span className="ml-3">{item.name}</span>}
            </Link>
          )
        })}

        {/* Collapsible "More" — Monitors, Analysis (expanded) or icon-only (collapsed) */}
        {isOpen ? (
          <div className="pt-2 mt-2 border-t border-slate-200/60">
            <button
              type="button"
              onClick={() => setShowMoreNav(!showMoreNav)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-50 transition-colors text-left"
              aria-expanded={showMoreNav}
            >
              <span className="text-sm font-medium">More</span>
              {showMoreNav ? (
                <ChevronUpIcon className="w-4 h-4" />
              ) : (
                <ChevronDownIcon className="w-4 h-4" />
              )}
            </button>
            {showMoreNav && (
              <div className="space-y-1 mt-1 pl-1">
                {moreNav.map((item) => {
                  const isActive = location.pathname === item.href || location.pathname.startsWith(item.href + '/')
                  const Icon = item.icon
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
                      aria-current={isActive ? 'page' : undefined}
                    >
                      <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-primary-600' : 'text-slate-400'}`} />
                      <span className="ml-3">{item.name}</span>
                    </Link>
                  )
                })}
              </div>
            )}
          </div>
        ) : (
          /* Collapsed: show Monitors & Analysis as icon-only */
          <div className="pt-2 mt-2 border-t border-slate-200/60 space-y-1">
            {moreNav.map((item) => {
              const isActive = location.pathname === item.href || location.pathname.startsWith(item.href + '/')
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`nav-link ${isActive ? 'nav-link-active' : 'nav-link-inactive'}`}
                  title={item.name}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-primary-600' : 'text-slate-400'}`} />
                </Link>
              )
            })}
          </div>
        )}
      </nav>

      {/* Connection Status Footer */}
      <div className="px-3 py-4 border-t border-slate-200/80 space-y-3">
        {isOpen ? (
          <div className="px-3 py-2.5 bg-slate-50 rounded-xl border border-slate-200/60">
            <div className="flex items-center space-x-2 mb-1">
              <div className={`w-2 h-2 rounded-full ${getStatusColor()} ${connectionStatus?.is_active ? 'animate-pulse' : ''}`} />
              <span className="text-xs font-medium text-slate-700">{getStatusText()}</span>
            </div>
            {connectionStatus?.company_name && (
              <div className="text-xs text-slate-600 truncate">{connectionStatus.company_name}</div>
            )}
          </div>
        ) : (
          <div className="flex justify-center">
            <div className={`w-3 h-3 rounded-full ${getStatusColor()} ${connectionStatus?.is_active ? 'animate-pulse' : ''}`} />
          </div>
        )}
        {isOpen && (
          <div className="text-center text-xs text-slate-400">
            v0.1.0
          </div>
        )}
      </div>
    </div>
  )
}
