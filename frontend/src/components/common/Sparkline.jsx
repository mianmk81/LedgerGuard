import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function Sparkline({ data, color = '#0ea5e9', className = '' }) {
  const chartData = data.map((value, index) => ({ value, index }))

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
