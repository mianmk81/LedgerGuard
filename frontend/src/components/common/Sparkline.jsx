import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function Sparkline({ data, color = '#0284c7', className = '' }) {
  if (!data || !data.length) return null

  const chartData = data.map((value, index) => ({ value, index }))

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            strokeOpacity={0.9}
            dot={false}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
