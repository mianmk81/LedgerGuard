import { useState, useEffect, useCallback, useRef } from 'react'
import { Bell, User, LogOut, RefreshCw, ChevronDown, Menu } from 'lucide-react'
import { api } from '../../api/client'

export default function Header({ onLogout, onMenuToggle }) {
  const [connectionStatus, setConnectionStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)
  const menuButtonRef = useRef(null)

  const fetchConnectionStatus = useCallback(async () => {
    try {
      const status = await api.connection.getStatus()
      setConnectionStatus(status)
    } catch (error) {
      console.error('Failed to fetch connection status:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchConnectionStatus()
  }, [fetchConnectionStatus])

  useEffect(() => {
    if (!menuOpen) return
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false)
      }
    }
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        setMenuOpen(false)
        menuButtonRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [menuOpen])

  return (
    <header className="h-16 bg-white/80 backdrop-blur-xl border-b border-slate-200/60 flex items-center justify-between px-4 md:px-6 shadow-soft sticky top-0 z-40">
      <div className="flex items-center space-x-3">
        {onMenuToggle && (
          <button
            onClick={onMenuToggle}
            className="p-2 rounded-xl hover:bg-slate-100 active:bg-slate-200 transition-all duration-200 md:hidden"
            aria-label="Open navigation menu"
          >
            <Menu className="w-5 h-5 text-slate-600" />
          </button>
        )}
        {!loading && connectionStatus && (
          <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-emerald-50 to-emerald-100/50 border border-emerald-300/60 rounded-full">
            <div className="relative">
              <div className="absolute inset-0 bg-emerald-400 rounded-full blur-sm opacity-75" />
              <div className="relative w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            </div>
            <span className="text-sm font-bold text-emerald-900">
              {connectionStatus.company_name || 'Connected'}
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center space-x-1">
        <button
          onClick={fetchConnectionStatus}
          className="p-2.5 rounded-xl hover:bg-slate-100 active:bg-slate-200 transition-all duration-200"
          aria-label="Refresh connection status"
        >
          <RefreshCw className="w-5 h-5 text-slate-500" />
        </button>

        <button
          className="p-2.5 rounded-xl hover:bg-slate-100 active:bg-slate-200 transition-all duration-200 relative"
          aria-label="Notifications (coming soon)"
          title="Notifications coming soon"
        >
          <Bell className="w-5 h-5 text-slate-500" />
        </button>

        {/* User Menu — keyboard accessible */}
        <div className="relative" ref={menuRef}>
          <button
            ref={menuButtonRef}
            onClick={() => setMenuOpen((prev) => !prev)}
            className="flex items-center space-x-2 px-3 py-2 rounded-xl hover:bg-slate-100 active:bg-slate-200 transition-all duration-200"
            aria-expanded={menuOpen}
            aria-haspopup="true"
            aria-label="User menu"
          >
            <div className="w-8 h-8 bg-primary-100 rounded-xl flex items-center justify-center ring-1 ring-primary-200/50">
              <User className="w-5 h-5 text-primary-600" />
            </div>
            <span className="text-sm font-medium text-slate-700">Admin</span>
            <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${menuOpen ? 'rotate-180' : ''}`} />
          </button>

          {menuOpen && (
            <div
              className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-hover border border-slate-200/80 py-2 animate-fade-in"
              role="menu"
            >
              <button
                onClick={() => { onLogout(); setMenuOpen(false) }}
                className="w-full flex items-center space-x-2 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
                role="menuitem"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
