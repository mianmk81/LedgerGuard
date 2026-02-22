import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  Banknote,
  FileText,
  MessageSquare,
  Receipt,
  AlertCircle,
  CheckSquare,
  TrendingUp,
  Calendar,
  Zap,
  Repeat,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

const RISK_BUCKET_STYLES = {
  low: 'text-green-600 bg-green-50 border-green-200',
  medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
  high: 'text-orange-600 bg-orange-50 border-orange-200',
  critical: 'text-red-600 bg-red-50 border-red-200',
  escalating_frustration: 'text-red-600 bg-red-50 border-red-200',
}

const RISK_BUCKET_LABELS = {
  low: 'Low risk — looking good, no action needed right now',
  medium: 'Moderate risk — worth keeping an eye on',
  high: 'High risk — likely to cause problems if not addressed soon',
  critical: 'Critical risk — needs immediate attention',
  escalating_frustration: 'Customers are getting increasingly frustrated',
}

export default function Insights() {
  const [loading, setLoading] = useState(true)
  const [cashRunway, setCashRunway] = useState(null)
  const [invoiceRisk, setInvoiceRisk] = useState(null)
  const [ticketSentiment, setTicketSentiment] = useState(null)
  const [taxLiability, setTaxLiability] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [cashForecast, setCashForecast] = useState(null)
  const [followUpPriorities, setFollowUpPriorities] = useState(null)
  const [periodComparison, setPeriodComparison] = useState(null)
  const [scenarioExpense, setScenarioExpense] = useState(null)
  const [scenarioInvoice, setScenarioInvoice] = useState(null)
  const [recurringPatterns, setRecurringPatterns] = useState(null)
  const [error, setError] = useState(null)
  const [showMoreInsights, setShowMoreInsights] = useState(false)

  const fetchAll = useCallback(async () => {
    try {
      setError(null)
      const [
        runway, invRisk, sentiment, tax,
        recs, forecast, followUp, periodComp,
        scenExp, scenInv, patterns,
      ] = await Promise.all([
        api.insights.cashRunway(60).catch(() => null),
        api.insights.invoiceDefaultRisk({ top_n: 10 }).catch(() => null),
        api.insights.supportTicketSentiment({ top_n: 20 }).catch(() => null),
        api.insights.taxLiabilityEstimate({ lookback_days: 365 }).catch(() => null),
        api.insights.recommendations(5).catch(() => null),
        api.insights.cashForecastCurve(6, 60).catch(() => null),
        api.insights.invoiceFollowUpPriorities(5).catch(() => null),
        api.insights.periodComparison('month').catch(() => null),
        api.insights.scenarioExpense(10).catch(() => null),
        api.insights.scenarioInvoiceCollection(5).catch(() => null),
        api.insights.recurringPatterns(365).catch(() => null),
      ])
      setCashRunway(runway)
      setInvoiceRisk(invRisk)
      setTicketSentiment(sentiment)
      setTaxLiability(tax)
      setRecommendations(recs)
      setCashForecast(forecast)
      setFollowUpPriorities(followUp)
      setPeriodComparison(periodComp)
      setScenarioExpense(scenExp)
      setScenarioInvoice(scenInv)
      setRecurringPatterns(patterns)
    } catch (err) {
      console.error('Failed to fetch insights:', err)
      setError(err.message || 'Failed to load insights')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  if (loading) {
    return <LoadingState message="Loading business insights..." />
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-gray-900">Business Insights</h1>
        <div className="card border-red-200 bg-red-50/50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-8 h-8 text-red-500 flex-shrink-0" />
            <div>
              <p className="font-medium text-red-800">Failed to load insights</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <button onClick={() => { setLoading(true); fetchAll(); }} className="mt-3 btn btn-secondary text-sm">
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Business Insights</h1>
        <p className="text-gray-600 mt-1">
          Cash runway, invoice risk, ticket sentiment, and tax estimates
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* H4: Cash Runway */}
        <div className="card-interactive">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-blue-50 flex items-center justify-center">
              <Banknote className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Cash Runway</h2>
              <p className="text-sm text-gray-500">Months until cash runs out</p>
            </div>
          </div>
          {cashRunway ? (
            <div className="space-y-2">
              {cashRunway.runway_infinite ? (
                <p className="text-lg font-medium text-green-600">Positive cash flow — runway infinite</p>
              ) : cashRunway.runway_months != null ? (
                <p className="text-lg font-medium text-gray-900">
                  {cashRunway.runway_months.toFixed(1)} months
                  {cashRunway.runway_months_ci_lower != null && (
                    <span className="text-sm text-gray-500 ml-2">
                      (95% CI: {cashRunway.runway_months_ci_lower?.toFixed(1)}–{cashRunway.runway_months_ci_upper?.toFixed(1)})
                    </span>
                  )}
                </p>
              ) : (
                <p className="text-gray-600">Insufficient data</p>
              )}
              <p className="text-sm text-gray-500">
                Net cash proxy: ${cashRunway.net_cash_proxy?.toLocaleString?.() ?? 0} | Monthly burn: ${cashRunway.monthly_burn_rate?.toLocaleString?.() ?? 0}
              </p>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No data available</p>
          )}
        </div>

        {/* H5: Invoice Default Risk */}
        <div className="card-interactive">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center">
              <FileText className="w-5 h-5 text-amber-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Invoice Default Risk</h2>
              <p className="text-sm text-gray-500">Top receivables by default/late risk</p>
            </div>
          </div>
          {invoiceRisk?.receivables?.length ? (
            <div className="space-y-2">
              <p className="text-sm text-gray-600">
                ${invoiceRisk.summary?.total_amount_at_risk?.toLocaleString?.() ?? 0} at risk · {invoiceRisk.summary?.high_risk_count ?? 0} high/critical
              </p>
              <div className="max-h-48 overflow-y-auto space-y-1 text-sm">
                {invoiceRisk.receivables.slice(0, 5).map((r, i) => (
                  <div key={i} className="flex justify-between items-center py-1 border-b border-gray-100 last:border-0">
                    <span className="truncate flex-1">${r.amount?.toLocaleString?.() ?? 0}</span>
                    <span title={RISK_BUCKET_LABELS[r.risk_bucket] || ''} className={`px-2 py-0.5 rounded text-xs font-medium border cursor-help ${RISK_BUCKET_STYLES[r.risk_bucket] || ''}`}>
                      {r.risk_bucket?.replace(/_/g, ' ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No receivables to score</p>
          )}
        </div>

        {/* H6: Support Ticket Sentiment */}
        <div className="card-interactive">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-purple-50 flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Ticket Sentiment</h2>
              <p className="text-sm text-gray-500">Escalating frustration risk</p>
            </div>
          </div>
          {ticketSentiment?.tickets?.length ? (
            <div className="space-y-2">
              <p className="text-sm text-gray-600">
                {ticketSentiment.summary?.escalating_frustration_count ?? 0} escalating · {ticketSentiment.summary?.high_risk_count ?? 0} high risk
              </p>
              <div className="max-h-48 overflow-y-auto space-y-1 text-sm">
                {ticketSentiment.tickets.slice(0, 5).map((t, i) => (
                  <div key={i} className="py-1 border-b border-gray-100 last:border-0">
                    <p className="truncate text-gray-800">{t.subject}</p>
                    <span title={RISK_BUCKET_LABELS[t.risk_bucket] || ''} className={`inline-block mt-0.5 px-2 py-0.5 rounded text-xs font-medium border cursor-help ${RISK_BUCKET_STYLES[t.risk_bucket] || ''}`}>
                      {t.risk_bucket?.replace(/_/g, ' ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No tickets to analyze</p>
          )}
        </div>

        {/* H7: Tax Liability Estimate */}
        <div className="card-interactive">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center">
              <Receipt className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Tax Liability Estimate</h2>
              <p className="text-sm text-gray-500">From P&L + audit risk</p>
            </div>
          </div>
          {taxLiability ? (
            <div className="space-y-2">
              <p className="text-lg font-medium text-gray-900">
                Estimated: ${taxLiability.estimated_tax_liability?.toLocaleString?.() ?? 0}
              </p>
              <p className="text-sm text-gray-500">
                Taxable income: ${taxLiability.taxable_income_proxy?.toLocaleString?.() ?? 0} · Rate: {(taxLiability.effective_rate ?? 0) * 100}%
              </p>
              {taxLiability.audit_risk && taxLiability.audit_risk !== 'low' && (
                <div className="mt-2 p-2 rounded bg-amber-50 border border-amber-200">
                  <p className="text-sm font-medium text-amber-800">Audit risk: {taxLiability.audit_risk}</p>
                  {taxLiability.audit_factors?.length > 0 && (
                    <ul className="text-xs text-amber-700 mt-1 list-disc list-inside">
                      {taxLiability.audit_factors.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No data available</p>
          )}
        </div>
      </div>

      {/* Extended Features — Collapsed by default for hackathon demo */}
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        <button
          type="button"
          onClick={() => setShowMoreInsights(!showMoreInsights)}
          className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
        >
          <span className="font-medium text-gray-700">
            More insights — Recommendations, forecast curve, scenario planning, recurring patterns
          </span>
          {showMoreInsights ? (
            <ChevronUp className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-500" />
          )}
        </button>
        {showMoreInsights && (
        <div className="p-4 pt-0 border-t border-gray-200">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-4">
          {/* Recommendation Feed */}
          <div className="card-interactive lg:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-indigo-50 flex items-center justify-center">
                <CheckSquare className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Top Actions to Improve Score</h2>
                <p className="text-sm text-gray-500">Prioritized checklist from health + incidents</p>
              </div>
            </div>
            {recommendations?.recommendations?.length ? (
              <ul className="space-y-2">
                {recommendations.recommendations.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 py-2 border-b border-gray-100 last:border-0">
                    <span className="flex-shrink-0 w-5 h-5 rounded border border-gray-300 flex items-center justify-center text-xs">□</span>
                    <span className="text-sm text-gray-800">{r.action}</span>
                    {r.source && <span className="text-xs text-gray-400">({r.source})</span>}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500 text-sm">No recommendations yet. Run analysis to get suggestions.</p>
            )}
          </div>

          {/* Invoice Follow-Up Priorities */}
          <div className="card-interactive">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center">
                <FileText className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Follow Up Today</h2>
                <p className="text-sm text-gray-500">{followUpPriorities?.action_label ?? 'Top 5 to chase'}</p>
              </div>
            </div>
            {followUpPriorities?.receivables?.length ? (
              <div className="space-y-1">
                {followUpPriorities.receivables.slice(0, 5).map((r, i) => (
                  <div key={i} className="flex justify-between items-center py-1 text-sm">
                    <span>${r.amount?.toLocaleString?.() ?? 0}</span>
                    <span title={RISK_BUCKET_LABELS[r.risk_bucket] || ''} className={`px-2 py-0.5 rounded text-xs cursor-help ${RISK_BUCKET_STYLES[r.risk_bucket] || ''}`}>
                      {r.risk_bucket?.replace(/_/g, ' ')}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No invoices to prioritize</p>
            )}
          </div>

          {/* Cash Forecast Curve */}
          <div className="card-interactive lg:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-blue-50 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Cash Flow Forecast</h2>
                <p className="text-sm text-gray-500">6‑month projection with best/worst bands</p>
              </div>
            </div>
            {cashForecast?.curve?.length ? (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={cashForecast.curve} margin={{ top: 12, right: 12, bottom: 8, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#d1d5db" />
                    <XAxis
                      dataKey="month"
                      tick={{ fontSize: 11, fill: '#64748b' }}
                      tickFormatter={(v) => v?.slice?.(0, 7) ?? v}
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={(v) => `$${Number(v).toLocaleString()}`} />
                    <Legend formatter={(value) => <span style={{ color: '#475569' }}>{value}</span>} />
                    <Area type="monotone" dataKey="cash_worst" stroke="#b91c1c" fill="#b91c1c" fillOpacity={0.2} name="Worst case" />
                    <Area type="monotone" dataKey="cash_baseline" stroke="#1d4ed8" fill="#1d4ed8" fillOpacity={0.2} name="Baseline" />
                    <Area type="monotone" dataKey="cash_best" stroke="#15803d" fill="#15803d" fillOpacity={0.2} name="Best case" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-gray-500 text-sm">Insufficient data for forecast</p>
            )}
          </div>

          {/* Period Comparison */}
          <div className="card-interactive">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-violet-50 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-violet-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Period Comparison</h2>
                <p className="text-sm text-gray-500">This month vs last month</p>
              </div>
            </div>
            {periodComparison ? (
              <div className="space-y-2">
                <p className="text-sm">
                  Health: {periodComparison.health_score?.current ?? '—'} vs {periodComparison.health_score?.previous ?? '—'}
                  {periodComparison.health_score?.change_pct != null && (
                    <span className={periodComparison.health_score.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {' '}({periodComparison.health_score.change_pct >= 0 ? '+' : ''}{periodComparison.health_score.change_pct}%)
                    </span>
                  )}
                </p>
                <p className="text-sm text-gray-600">{periodComparison.summary}</p>
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No comparison data</p>
            )}
          </div>

          {/* Scenario Planning */}
          <div className="card-interactive lg:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-cyan-50 flex items-center justify-center">
                <Zap className="w-5 h-5 text-cyan-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Scenario Planning</h2>
                <p className="text-sm text-gray-500">What-if: expense cut & invoice collection</p>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
                <p className="text-xs font-medium text-gray-500 mb-1">If you cut expenses by 10%</p>
                <p className="text-sm text-gray-800">{scenarioExpense?.narrative ?? '—'}</p>
              </div>
              <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
                <p className="text-xs font-medium text-gray-500 mb-1">If you collect top 5 invoices</p>
                <p className="text-sm text-gray-800">{scenarioInvoice?.narrative ?? '—'}</p>
              </div>
            </div>
          </div>

          {/* Recurring Patterns */}
          <div className="card-interactive">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-rose-50 flex items-center justify-center">
                <Repeat className="w-5 h-5 text-rose-600" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">Recurring Patterns</h2>
                <p className="text-sm text-gray-500">Seasonality in incidents</p>
              </div>
            </div>
            {recurringPatterns?.patterns?.length ? (
              <div className="space-y-2">
                {recurringPatterns.patterns.slice(0, 3).map((p, i) => (
                  <div key={i} className="p-2 rounded bg-rose-50/50 border border-rose-100">
                    <p className="text-sm font-medium text-gray-800">{p.pattern_description}</p>
                    <p className="text-xs text-gray-600 mt-0.5">{p.recommendation}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">{recurringPatterns?.summary ?? 'Not enough incident history'}</p>
            )}
          </div>
        </div>
        </div>
        )}
      </div>
    </div>
  )
}
