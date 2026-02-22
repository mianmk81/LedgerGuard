import clsx from 'clsx'

const severityConfig = {
  critical: {
    bg: 'bg-gradient-to-r from-red-500 to-red-600',
    text: 'text-white',
    border: 'border-red-600',
    shadow: 'shadow-glow-danger',
    dot: 'bg-white',
    label: 'Critical',
    tooltip: 'Needs immediate action — could seriously impact your business',
  },
  high: {
    bg: 'bg-gradient-to-r from-orange-400 to-orange-500',
    text: 'text-white',
    border: 'border-orange-500',
    shadow: 'shadow-sm',
    dot: 'bg-white',
    label: 'High',
    tooltip: 'Significant problem — should be addressed soon',
  },
  medium: {
    bg: 'bg-gradient-to-r from-yellow-400 to-yellow-500',
    text: 'text-gray-900',
    border: 'border-yellow-500',
    shadow: 'shadow-sm',
    dot: 'bg-gray-900',
    label: 'Medium',
    tooltip: 'Moderate issue — keep an eye on it and plan a fix',
  },
  low: {
    bg: 'bg-gradient-to-r from-green-400 to-green-500',
    text: 'text-white',
    border: 'border-green-500',
    shadow: 'shadow-glow-success',
    dot: 'bg-white',
    label: 'Low',
    tooltip: 'Minor issue — not urgent but worth noting',
  },
}

export default function SeverityBadge({ severity, className = '' }) {
  const config = severityConfig[severity] || severityConfig.low

  return (
    <span
      title={config.tooltip}
      className={clsx(
        'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border transition-all duration-200 cursor-help',
        config.bg,
        config.text,
        config.border,
        config.shadow,
        className
      )}
    >
      <span className={clsx('w-1.5 h-1.5 rounded-full animate-pulse', config.dot)} />
      {config.label}
    </span>
  )
}
