/**
 * LoadingFallback — Suspense fallback component.
 *
 * React-specialist agent: Suspense boundaries with meaningful
 * loading states for lazy-loaded page components.
 */
import { Loader2 } from 'lucide-react'

export default function LoadingFallback({ message = 'Loading...' }) {
  return (
    <div className="flex items-center justify-center min-h-[400px]" role="status" aria-live="polite">
      <div className="text-center">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mx-auto mb-3" aria-hidden="true" />
        <p className="text-sm text-gray-500">{message}</p>
      </div>
    </div>
  )
}
