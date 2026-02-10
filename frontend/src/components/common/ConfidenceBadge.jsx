import clsx from 'clsx'

export default function ConfidenceBadge({ confidence, className = '' }) {
  const percentage = Math.round(confidence * 100)

  const getColor = () => {
    if (confidence >= 0.9) return 'bg-green-100 text-green-800 border-green-200'
    if (confidence >= 0.7) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    return 'bg-red-100 text-red-800 border-red-200'
  }

  return (
    <span
      className={clsx(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        getColor(),
        className
      )}
    >
      {percentage}% confidence
    </span>
  )
}
