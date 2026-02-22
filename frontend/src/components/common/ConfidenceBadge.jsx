import clsx from 'clsx'

const CONFIDENCE_CONFIG = {
  very_high: { label: 'Very confident', pct: 95, tooltip: 'Multiple detection methods agree — almost certainly a real issue, not a false alarm', color: 'bg-green-100 text-green-800 border-green-200' },
  high: { label: 'Confident', pct: 85, tooltip: 'Strong evidence this is a real issue', color: 'bg-green-100 text-green-800 border-green-200' },
  medium: { label: 'Moderate confidence', pct: 65, tooltip: 'Could be real but may need more data to confirm', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
  low: { label: 'Low confidence', pct: 40, tooltip: 'Weak signal — might be noise, worth monitoring', color: 'bg-red-100 text-red-800 border-red-200' },
}

export default function ConfidenceBadge({ confidence, className = '' }) {
  if (typeof confidence === 'string') {
    const key = confidence.toLowerCase().replace(/\s+/g, '_')
    const config = CONFIDENCE_CONFIG[key] || CONFIDENCE_CONFIG.medium
    return (
      <span
        title={config.tooltip}
        className={clsx(
          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border cursor-help',
          config.color,
          className
        )}
      >
        {config.label}
      </span>
    )
  }

  const safeConfidence = confidence ?? 0
  const percentage = Math.round(safeConfidence * 100)

  const getColor = () => {
    if (safeConfidence >= 0.9) return 'bg-green-100 text-green-800 border-green-200'
    if (safeConfidence >= 0.7) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    return 'bg-red-100 text-red-800 border-red-200'
  }

  const getTooltip = () => {
    if (safeConfidence >= 0.9) return 'Very strong evidence this is a real issue'
    if (safeConfidence >= 0.7) return 'Good evidence this is real'
    return 'Weak signal — worth monitoring but may not be a real issue'
  }

  return (
    <span
      title={getTooltip()}
      className={clsx(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border cursor-help',
        getColor(),
        className
      )}
    >
      {percentage}% confidence
    </span>
  )
}
