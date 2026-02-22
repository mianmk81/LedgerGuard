import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import {
  ChartBarIcon,
  ShieldCheckIcon,
  BeakerIcon,
  ArrowTrendingUpIcon,
  XMarkIcon,
  MagnifyingGlassPlusIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

// ─── Hard-coded production metrics (shown even without model card JSON) ─────
const PRODUCTION_MODELS = {
  churn_telco: {
    model_name: 'LightGBM Churn Classifier',
    model_type: 'LightGBM',
    dataset: 'IBM Telco — 7,043 customers, 30 features',
    industry: 'Telecom',
    feature_count: 30,
    cv_folds: 5,
    badge: 'Industry Level',
    badge_color: 'emerald',
    metrics: {
      test: { f1: 0.6495, precision: 0.72, recall: 0.60, auc: 0.8509, pr_auc: 0.6608 },
      cv: {
        f1: { mean: 0.64, std: 0.015, ci95_half_width: 0.013 },
        roc_auc: { mean: 0.845, std: 0.012, ci95_half_width: 0.010 },
      },
    },
    extra: { decision_threshold: 0.443 },
    key_features: ['MonthlyCharges', 'tenure', 'TotalCharges', 'contract_encoded', 'at_risk_combo'],
    insight: 'Customers on month-to-month contracts with high monthly charges and low tenure are 3× more likely to churn.',
  },
  delivery_dataco: {
    model_name: 'XGBoost Delivery Risk Predictor',
    model_type: 'XGBoost',
    dataset: 'DataCo Supply Chain — 180K+ orders, 25 features',
    industry: 'Logistics',
    feature_count: 25,
    cv_folds: 5,
    badge: 'Industry Level',
    badge_color: 'blue',
    metrics: {
      test: { f1: 0.7537, precision: 0.72, recall: 0.79, auc: 0.8176 },
      cv: {
        f1: { mean: 0.748, std: 0.008, ci95_half_width: 0.007 },
        roc_auc: { mean: 0.814, std: 0.006, ci95_half_width: 0.005 },
      },
    },
    extra: { decision_threshold: 0.3354 },
    key_features: ['scheduled_shipping_days', 'shipping_mode_encoded', 'tight_schedule', 'high_risk_combo', 'profit_ratio'],
    insight: 'Orders with tight shipping schedules via Standard Class with low profit ratios have a 68% late delivery rate.',
  },
  sentiment: {
    model_name: 'LinearSVC Sentiment Classifier',
    model_type: 'LinearSVC (Calibrated)',
    dataset: 'FinancialPhraseBank — 4,840 sentences, 3 classes',
    industry: 'Finance NLP',
    feature_count: 20000,
    cv_folds: 5,
    badge: 'Industry Level',
    badge_color: 'violet',
    metrics: {
      test: { f1: 0.8842, precision: 0.89, recall: 0.88, auc: 0.97, accuracy: 0.9176 },
      cv: {
        f1: { mean: 0.882, std: 0.009, ci95_half_width: 0.008 },
        roc_auc: { mean: 0.968, std: 0.005, ci95_half_width: 0.004 },
      },
      per_class: { negative: 0.85, neutral: 0.96, positive: 0.84 },
    },
    extra: { classes: ['Negative', 'Neutral', 'Positive'], ngram_range: '1-4', max_features: 20000 },
    key_features: ['profit', 'loss', 'growth', 'decline', 'quarterly', 'revenue', 'expects', 'fell'],
    insight: 'Words like "profit", "growth", "expects" are the strongest positive signals. "Fell", "loss", "decline" dominate negative classification.',
  },
  trend: {
    model_name: 'LightGBM Trend Forecaster',
    model_type: 'LightGBM Regressor (×8 metrics)',
    dataset: 'Live Business Metrics — Gold Layer (DuckDB)',
    industry: 'Time Series',
    feature_count: 19,
    cv_folds: null,
    badge: 'Zero Overfitting',
    badge_color: 'amber',
    metrics: {
      test: { f1: null, auc: null },
      per_metric: {
        refund_rate: { train_mae: 0.111, test_mae: 0.113 },
        margin_proxy: { train_mae: 0.025, test_mae: 0.020 },
        delivery_delay_rate: { train_mae: 0.042, test_mae: 0.045 },
        ticket_backlog: { train_mae: 3.1, test_mae: 3.4 },
        churn_proxy: { train_mae: 0.008, test_mae: 0.009 },
      },
    },
    extra: { metrics_covered: 8, projection_days: 7, lag_features: 14 },
    key_features: ['lag_0 (yesterday)', 'lag_1', 'rolling_7_mean', 'rolling_14_mean', 'trend_delta'],
    insight: 'Train MAE ≈ Test MAE across all 8 metrics — no overfitting. Used for 7-day ahead projections in the Early Warning system.',
  },
}

const PRETRAINED_MODELS = [
  {
    name: 'Isolation Forest',
    type: 'Anomaly Detection',
    file: 'models/anomaly/isolation_forest.joblib',
    metric: 'CV F1: 0.72 ± 0.19',
    desc: 'Trained on 443 days of business metrics. Detects anomalies via random partitioning — anomalies require fewer splits to isolate.',
    color: 'red',
  },
  {
    name: 'One-Class SVM',
    type: 'Anomaly Detection',
    file: 'models/anomaly/one_class_svm.joblib',
    metric: 'Learns normal boundary',
    desc: 'RBF kernel support vector machine trained only on normal data. Flags points outside the learned normal region.',
    color: 'red',
  },
  {
    name: 'Local Outlier Factor',
    type: 'Anomaly Detection',
    file: 'models/anomaly/local_outlier_factor.joblib',
    metric: 'Density-based scoring',
    desc: 'Compares local density of each point to its 20 nearest neighbours. Low relative density = anomaly candidate.',
    color: 'red',
  },
  {
    name: 'Autoencoder (MLP)',
    type: 'Anomaly Detection',
    file: 'models/anomaly/autoencoder.joblib',
    metric: 'Reconstruction error',
    desc: 'Bottleneck architecture (128→64→32→64→128). High reconstruction error = anomaly. Threshold at 95th percentile of training errors.',
    color: 'red',
  },
  {
    name: 'LightGBM Churn (Olist)',
    type: 'Churn Prediction',
    file: 'models/churn/lightgbm_churn_model.pkl',
    metric: 'F1: 0.89, AUC: 0.994',
    desc: '17 RFM + delivery experience features on 2,381 Olist customers. High AUC reflects Olist\'s clear churn patterns.',
    color: 'blue',
  },
  {
    name: 'Random Forest Churn',
    type: 'Churn Prediction',
    file: 'models/churn/random_forest_churn_model.pkl',
    metric: 'Ensemble baseline',
    desc: '200 trees, max_depth=10. Trained alongside LightGBM as a comparison baseline on Olist churn data.',
    color: 'blue',
  },
  {
    name: 'XGBoost Delivery (Olist)',
    type: 'Delivery Risk',
    file: 'models/delivery/xgboost_late_delivery.joblib',
    metric: 'F1: 0.42, AUC: 0.83',
    desc: 'Zero-leakage model — no review_score, no actual dates. F1 0.42 reflects the hard 8% late rate problem with only order-time features.',
    color: 'green',
  },
  {
    name: 'Stacked Ensemble Delivery',
    type: 'Delivery Risk',
    file: 'models/delivery/stacked_ensemble_late_delivery.joblib',
    metric: 'F1: 0.42, AUC: 0.83',
    desc: 'XGBoost + RF + LR base learners with LogisticRegression meta-learner. OOF predictions prevent leakage.',
    color: 'green',
  },
  {
    name: 'TF-IDF + Logistic Regression',
    type: 'Sentiment Analysis',
    file: 'models/sentiment_industry/tfidf_lr_financial_sentiment.joblib',
    metric: 'Macro F1: 0.82',
    desc: 'Baseline model for financial sentiment. 10K features, bigrams. Used as fallback when LinearSVC unavailable.',
    color: 'violet',
  },
]

const TABS = [
  { id: 'churn_telco',     label: 'Churn',     subtitle: 'Telco LightGBM',    icon: ChartBarIcon,      color: 'emerald' },
  { id: 'delivery_dataco', label: 'Delivery',  subtitle: 'DataCo XGBoost',    icon: ShieldCheckIcon,   color: 'blue'    },
  { id: 'sentiment',       label: 'Sentiment', subtitle: 'LinearSVC',         icon: SparklesIcon,      color: 'violet'  },
  { id: 'trend',           label: 'Trend',     subtitle: 'LightGBM ×8',       icon: ArrowTrendingUpIcon, color: 'amber' },
]

const COLOR_MAP = {
  emerald: { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', badge: 'bg-emerald-100 text-emerald-800', tab: 'border-emerald-500 text-emerald-700' },
  blue:    { bg: 'bg-blue-50',    border: 'border-blue-200',    text: 'text-blue-700',    badge: 'bg-blue-100 text-blue-800',    tab: 'border-blue-500 text-blue-700'    },
  violet:  { bg: 'bg-violet-50',  border: 'border-violet-200',  text: 'text-violet-700',  badge: 'bg-violet-100 text-violet-800',tab: 'border-violet-500 text-violet-700' },
  amber:   { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   badge: 'bg-amber-100 text-amber-800',  tab: 'border-amber-500 text-amber-700'  },
  red:     { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-700',     badge: 'bg-red-100 text-red-800',     tab: 'border-red-500 text-red-700'      },
  green:   { bg: 'bg-green-50',   border: 'border-green-200',   text: 'text-green-700',   badge: 'bg-green-100 text-green-800', tab: 'border-green-500 text-green-700'  },
}

// ─── Image Lightbox ──────────────────────────────────────────────────────────
function ImageLightbox({ src, alt, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div className="relative max-w-5xl max-h-[90vh] w-full" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={onClose}
          className="absolute -top-10 right-0 text-white/80 hover:text-white flex items-center gap-1.5 text-sm"
        >
          <XMarkIcon className="w-5 h-5" /> Close (ESC)
        </button>
        <img
          src={src}
          alt={alt}
          className="w-full h-full max-h-[85vh] object-contain rounded-xl shadow-2xl bg-white"
        />
      </div>
    </div>
  )
}

// ─── Zoomable Image ──────────────────────────────────────────────────────────
function ZoomableImage({ src, alt, label }) {
  const [lightbox, setLightbox] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)

  return (
    <div>
      {label && <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">{label}</p>}
      <div
        className="relative group cursor-zoom-in rounded-xl border border-slate-200 bg-slate-50 overflow-hidden"
        onClick={() => !failed && setLightbox(true)}
      >
        {!failed ? (
          <>
            <img
              src={src}
              alt={alt}
              className={`w-full object-contain bg-white transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
              style={{ maxHeight: '220px' }}
              onLoad={() => setLoaded(true)}
              onError={() => setFailed(true)}
            />
            {!loaded && (
              <div className="h-40 flex items-center justify-center">
                <div className="w-6 h-6 border-2 border-slate-300 border-t-primary-500 rounded-full animate-spin" />
              </div>
            )}
            {loaded && (
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                <MagnifyingGlassPlusIcon className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
              </div>
            )}
          </>
        ) : (
          <div className="h-32 flex flex-col items-center justify-center text-slate-400 text-sm gap-1">
            <BeakerIcon className="w-6 h-6" />
            <span>Run generate scripts to create this plot</span>
          </div>
        )}
      </div>
      {lightbox && <ImageLightbox src={src} alt={alt} onClose={() => setLightbox(false)} />}
    </div>
  )
}

// ─── Metric Pill ─────────────────────────────────────────────────────────────
function MetricPill({ label, value, sub, highlight }) {
  return (
    <div className={`rounded-xl p-4 ${highlight ? 'bg-primary-50 border border-primary-100' : 'bg-slate-50 border border-slate-100'}`}>
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${highlight ? 'text-primary-700' : 'text-slate-800'}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

// ─── Production Model Card ───────────────────────────────────────────────────
function ProductionModelCard({ modelId, card: apiCard, reportBaseUrl }) {
  const defaults = PRODUCTION_MODELS[modelId]
  const card = apiCard || defaults
  const colors = COLOR_MAP[defaults.badge_color]
  const m = card.metrics || defaults.metrics

  const imgKey = {
    churn_telco: 'churn_telco',
    delivery_dataco: 'delivery_dataco',
    sentiment: 'sentiment',
    trend: 'trend',
  }[modelId]

  const showROC = modelId === 'churn_telco' || modelId === 'delivery_dataco'

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className={`rounded-2xl border ${colors.border} ${colors.bg} p-5`}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full ${colors.badge}`}>
                {defaults.badge}
              </span>
              <span className="text-xs text-slate-500">{defaults.industry}</span>
            </div>
            <h2 className="text-xl font-bold text-slate-800">{defaults.model_name}</h2>
            <p className="text-sm text-slate-600 mt-0.5">{defaults.model_type}</p>
            <p className="text-xs text-slate-500 mt-1">{defaults.dataset}</p>
          </div>
          <div className="text-right shrink-0">
            <p className="text-xs text-slate-500">Features</p>
            <p className="text-2xl font-bold text-slate-700">{defaults.feature_count.toLocaleString()}</p>
            {defaults.cv_folds && <p className="text-xs text-slate-500">{defaults.cv_folds}-fold CV</p>}
          </div>
        </div>

        {/* Key insight */}
        <div className="mt-4 flex items-start gap-2 bg-white/70 rounded-xl p-3 border border-white/80">
          <SparklesIcon className={`w-4 h-4 ${colors.text} shrink-0 mt-0.5`} />
          <p className="text-sm text-slate-700">{defaults.insight}</p>
        </div>
      </div>

      {/* Metrics row */}
      {modelId !== 'trend' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricPill
            label="Test F1"
            value={m.test?.f1 != null ? (m.test.f1 * 100).toFixed(1) + '%' : '—'}
            highlight
          />
          <MetricPill
            label="CV F1 ± 95% CI"
            value={m.cv?.f1?.mean != null
              ? (m.cv.f1.mean * 100).toFixed(1) + '%'
              : '—'}
            sub={m.cv?.f1?.ci95_half_width != null
              ? `± ${(m.cv.f1.ci95_half_width * 100).toFixed(1)}%`
              : null}
          />
          <MetricPill
            label="ROC-AUC"
            value={m.test?.auc != null ? m.test.auc.toFixed(4) : '—'}
          />
          <MetricPill
            label="Precision / Recall"
            value={m.test?.precision != null
              ? `${(m.test.precision * 100).toFixed(0)} / ${(m.test.recall * 100).toFixed(0)}`
              : '—'}
            sub="percent"
          />
        </div>
      )}

      {/* Trend MAE table */}
      {modelId === 'trend' && (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
          <div className="px-5 py-3 bg-slate-50 border-b border-slate-100">
            <p className="text-sm font-semibold text-slate-700">Per-Metric MAE (Train ≈ Test = No Overfitting)</p>
          </div>
          <div className="divide-y divide-slate-100">
            {Object.entries(defaults.metrics.per_metric).map(([metric, vals]) => (
              <div key={metric} className="flex items-center justify-between px-5 py-3">
                <span className="text-sm text-slate-700 font-mono">{metric}</span>
                <div className="flex gap-6 text-sm">
                  <span className="text-slate-500">Train: <span className="font-medium text-slate-800">{vals.train_mae}</span></span>
                  <span className="text-slate-500">Test: <span className="font-medium text-emerald-700">{vals.test_mae}</span></span>
                  <span className={`text-xs font-semibold ${Math.abs(vals.test_mae - vals.train_mae) / vals.train_mae < 0.1 ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {Math.abs(vals.test_mae - vals.train_mae) / vals.train_mae < 0.1 ? '✓ Stable' : '⚠ Check'}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-500">
            8 separate LightGBM regressors · 14 lag features + 5 rolling stats · 7-day projection horizon
          </div>
        </div>
      )}

      {/* Sentiment per-class metrics */}
      {modelId === 'sentiment' && m.per_class && (
        <div className="grid grid-cols-3 gap-3">
          {[['Negative', m.per_class.negative, 'red'], ['Neutral', m.per_class.neutral, 'slate'], ['Positive', m.per_class.positive, 'emerald']].map(([cls, f1, col]) => (
            <div key={cls} className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-center">
              <p className="text-xs font-medium text-slate-500">{cls}</p>
              <p className="text-2xl font-bold text-slate-800 mt-1">{(f1 * 100).toFixed(1)}%</p>
              <p className="text-xs text-slate-500">F1</p>
            </div>
          ))}
        </div>
      )}

      {/* Threshold callout */}
      {defaults.extra?.decision_threshold && (
        <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm">
          <span className="font-medium text-slate-600">Optimal Decision Threshold:</span>
          <span className="font-bold text-primary-700">{defaults.extra.decision_threshold}</span>
          <span className="text-slate-500 text-xs">(Optuna F1-tuned on validation set)</span>
        </div>
      )}

      {/* Key features */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <p className="text-sm font-semibold text-slate-700 mb-3">Top Features</p>
        <div className="flex flex-wrap gap-2">
          {defaults.key_features.map((f, i) => (
            <span key={f} className={`text-xs px-3 py-1.5 rounded-full font-medium ${
              i === 0 ? `${colors.badge} ring-1 ${colors.border}` : 'bg-slate-100 text-slate-700'
            }`}>
              #{i + 1} {f}
            </span>
          ))}
        </div>
      </div>

      {/* SHAP + CV plots */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <BeakerIcon className="w-4 h-4" />
          SHAP Explainability
          <span className="text-xs text-slate-400 font-normal">— click any image to zoom</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ZoomableImage src={reportBaseUrl(`shap_summary_${imgKey}.png`)} alt="SHAP summary" label="Feature Impact (Beeswarm)" />
          <ZoomableImage src={reportBaseUrl(`shap_bar_${imgKey}.png`)} alt="SHAP bar" label="Mean |SHAP| Ranking" />
          <ZoomableImage src={reportBaseUrl(`shap_waterfall_${imgKey}.png`)} alt="SHAP waterfall" label="Single Prediction Decomposition" />
          <ZoomableImage src={reportBaseUrl(`confusion_matrix_${imgKey}.png`)} alt="Confusion matrix" label="Confusion Matrix" />
        </div>
      </div>

      {/* CV + ROC */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
          <ChartBarIcon className="w-4 h-4" />
          Statistical Rigor
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ZoomableImage src={reportBaseUrl(`cv_boxplot_${imgKey}.png`)} alt="CV box plot" label="5-Fold CV Distribution" />
          {showROC && <ZoomableImage src={reportBaseUrl(`roc_curve_${imgKey}.png`)} alt="ROC curve" label="ROC Curve (Bootstrapped 95% CI)" />}
          {modelId === 'trend' && <ZoomableImage src={reportBaseUrl('trend_forecaster_mae_comparison.png')} alt="MAE comparison" label="Train vs Test MAE (All 8 Metrics)" />}
        </div>
      </div>
    </div>
  )
}

// ─── Data Integrity Card (Delivery) ─────────────────────────────────────────
function DataIntegrityCard() {
  return (
    <div className="rounded-2xl border border-amber-200 bg-gradient-to-br from-amber-50 to-orange-50 p-5 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <ShieldCheckIcon className="w-5 h-5 text-amber-700" />
        <h3 className="font-bold text-amber-900">Data Integrity: We Chose Correctness Over Inflated Metrics</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="rounded-xl bg-red-50 border border-red-200 p-4">
          <p className="text-xs font-bold text-red-700 uppercase mb-2">❌ With Data Leakage</p>
          <p className="text-3xl font-black text-red-600">~0.85</p>
          <p className="text-sm text-red-700 mt-1">F1 Score</p>
          <ul className="mt-3 text-xs text-red-600 space-y-1">
            <li>✗ Uses <code className="bg-red-100 px-1 rounded">review_score</code> (written after delivery)</li>
            <li>✗ Uses <code className="bg-red-100 px-1 rounded">actual_delivery_date</code> (unknown at order time)</li>
            <li>✗ Train/test split ignores time ordering</li>
          </ul>
        </div>
        <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4">
          <p className="text-xs font-bold text-emerald-700 uppercase mb-2">✓ Zero Leakage (Ours)</p>
          <div className="flex gap-4">
            <div>
              <p className="text-3xl font-black text-emerald-700">0.75</p>
              <p className="text-sm text-emerald-700">F1 Score</p>
            </div>
            <div>
              <p className="text-3xl font-black text-emerald-700">0.82</p>
              <p className="text-sm text-emerald-700">ROC-AUC</p>
            </div>
          </div>
          <ul className="mt-3 text-xs text-emerald-700 space-y-1">
            <li>✓ Only order-time features (known before dispatch)</li>
            <li>✓ Temporal train/test split (no future in training)</li>
            <li>✓ Seller performance history computed before cutoff</li>
          </ul>
        </div>
      </div>
      <p className="text-sm text-amber-800 font-medium">
        ROC-AUC 0.82 means the model correctly ranks 82% of late vs on-time orders.
        The F1 reflects the real-world challenge, not a modeling failure.
      </p>
    </div>
  )
}

// ─── Pre-trained Models Showcase ─────────────────────────────────────────────
function PretrainedShowcase() {
  const [expanded, setExpanded] = useState(false)
  const groups = {
    'Anomaly Detection Ensemble': PRETRAINED_MODELS.filter(m => m.type === 'Anomaly Detection'),
    'Churn (Olist Research)': PRETRAINED_MODELS.filter(m => m.type === 'Churn Prediction'),
    'Delivery Risk (Olist Research)': PRETRAINED_MODELS.filter(m => m.type === 'Delivery Risk'),
    'Sentiment (Baseline)': PRETRAINED_MODELS.filter(m => m.type === 'Sentiment Analysis'),
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-6 py-5 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <CpuChipIcon className="w-5 h-5 text-slate-500" />
          <div className="text-left">
            <p className="font-semibold text-slate-800">Pre-trained Research Models</p>
            <p className="text-sm text-slate-500">{PRETRAINED_MODELS.length} models — anomaly ensemble, Olist churn/delivery, baseline sentiment</p>
          </div>
        </div>
        <span className="text-slate-400 text-sm">{expanded ? '▲ Collapse' : '▼ Explore'}</span>
      </button>

      {expanded && (
        <div className="border-t border-slate-100 p-6 space-y-6">
          {/* Anomaly ensemble explanation */}
          <div className="rounded-xl bg-gradient-to-br from-red-50 to-rose-50 border border-red-100 p-5">
            <div className="flex items-center gap-2 mb-3">
              <BeakerIcon className="w-5 h-5 text-red-600" />
              <h3 className="font-bold text-red-900">4-Model Anomaly Ensemble — How It Works</h3>
            </div>
            <p className="text-sm text-red-800 mb-4">
              All 4 models vote independently on each day's 36-metric business snapshot.
              Requires 3/4 agreement (majority) to flag an anomaly — balancing precision and recall.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'Autoencoder'].map((m, i) => (
                <div key={m} className="rounded-lg bg-white border border-red-100 p-3 text-center text-xs">
                  <p className="font-semibold text-red-800">Model {i+1}</p>
                  <p className="text-red-600 mt-0.5">{m}</p>
                  <p className="text-slate-500 mt-1">↓ Vote</p>
                </div>
              ))}
            </div>
            <div className="mt-3 text-center text-sm text-red-700 font-medium">
              3 of 4 agree → ANOMALY flagged with VERY_HIGH confidence
            </div>
          </div>

          {/* Model grid by group */}
          {Object.entries(groups).map(([group, models]) => (
            <div key={group}>
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">{group}</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {models.map((m) => {
                  const c = COLOR_MAP[m.color] || COLOR_MAP.blue
                  return (
                    <div key={m.name} className={`rounded-xl border ${c.border} ${c.bg} p-4`}>
                      <div className="flex items-start justify-between gap-2">
                        <p className="font-semibold text-slate-800 text-sm">{m.name}</p>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${c.badge} shrink-0`}>{m.metric}</span>
                      </div>
                      <p className="text-xs text-slate-600 mt-2">{m.desc}</p>
                      <p className="text-xs text-slate-400 mt-2 font-mono">{m.file}</p>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── MLflow Panel ────────────────────────────────────────────────────────────
function MLflowPanel({ experiments }) {
  if (!experiments || experiments.length === 0) return null
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
      <div className="flex items-center gap-2 mb-4">
        <ChartBarIcon className="w-5 h-5 text-slate-500" />
        <h2 className="font-semibold text-slate-800">MLflow Experiment Tracking</h2>
        <span className="ml-auto text-xs text-slate-400 font-mono">mlflow ui → http://localhost:5000</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {experiments.map((exp) => (
          <div key={exp.experiment} className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${exp.status === 'active' ? 'bg-emerald-400' : 'bg-slate-300'}`} />
              <p className="font-medium text-slate-800 text-sm">{exp.experiment}</p>
            </div>
            <p className="text-xs text-slate-500 mb-3">{exp.run_count} runs tracked</p>
            {exp.best_run?.metrics && (
              <div className="space-y-1.5">
                {Object.entries(exp.best_run.metrics)
                  .filter(([k]) => k.includes('f1') || k.includes('auc') || k === 'accuracy')
                  .slice(0, 3)
                  .map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-slate-500">{k}</span>
                      <span className="font-semibold text-slate-800">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                    </div>
                  ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Main Page ───────────────────────────────────────────────────────────────
export default function ModelPerformance() {
  const [activeTab, setActiveTab] = useState('churn_telco')
  const [loading, setLoading] = useState(true)
  const [modelCards, setModelCards] = useState({})
  const [experiments, setExperiments] = useState([])

  const reportBaseUrl = useCallback(
    (filename) => `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/v1/system/reports/${filename}`,
    []
  )

  useEffect(() => {
    async function fetchData() {
      setLoading(true)
      try {
        const [cardsRes, experimentsRes] = await Promise.all([
          api.system.modelCards().catch(() => null),
          api.system.experiments().catch(() => null),
        ])
        setModelCards(cardsRes?.cards ?? cardsRes?.data?.cards ?? {})
        setExperiments(experimentsRes?.experiments ?? experimentsRes?.data?.experiments ?? [])
      } catch (_) {}
      setLoading(false)
    }
    fetchData()
  }, [])

  if (loading) return <LoadingState message="Loading model performance data..." />

  const activeTabConfig = TABS.find(t => t.id === activeTab)
  const activeColors = COLOR_MAP[activeTabConfig?.color || 'blue']

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Model Performance</h1>
        <p className="text-slate-500 mt-1 text-sm">
          4 production models · SHAP explainability · 5-fold CV · MLflow tracking · Click any chart to zoom
        </p>
      </div>

      {/* Summary bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: 'Churn AUC-ROC', value: '0.851', sub: 'Telco LightGBM', color: 'emerald' },
          { label: 'Delivery F1', value: '75.4%', sub: 'DataCo XGBoost', color: 'blue' },
          { label: 'Sentiment Accuracy', value: '91.8%', sub: 'LinearSVC Macro F1 0.884', color: 'violet' },
          { label: 'Trend MAE', value: '0.113', sub: 'refund_rate · zero overfit', color: 'amber' },
        ].map(({ label, value, sub, color }) => {
          const c = COLOR_MAP[color]
          return (
            <div key={label} className={`rounded-xl border ${c.border} ${c.bg} p-4`}>
              <p className="text-xs font-medium text-slate-500">{label}</p>
              <p className={`text-2xl font-black ${c.text} mt-1`}>{value}</p>
              <p className="text-xs text-slate-500 mt-0.5">{sub}</p>
            </div>
          )
        })}
      </div>

      {/* Tab bar */}
      <div className="border-b border-slate-200">
        <nav className="flex gap-1">
          {TABS.map(({ id, label, subtitle, icon: Icon, color }) => {
            const c = COLOR_MAP[color]
            const isActive = activeTab === id
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-2 py-3 px-4 border-b-2 font-medium text-sm transition-all rounded-t-lg ${
                  isActive
                    ? `${c.tab} bg-white`
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
                <span className={`hidden md:inline text-xs ${isActive ? 'opacity-70' : 'opacity-50'}`}>{subtitle}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab content */}
      <div>
        {activeTab === 'delivery_dataco' && <div className="mb-5"><DataIntegrityCard /></div>}
        <ProductionModelCard
          modelId={activeTab}
          card={modelCards?.[activeTab]}
          reportBaseUrl={reportBaseUrl}
        />
      </div>

      {/* MLflow panel */}
      <MLflowPanel experiments={experiments} />

      {/* Pre-trained models */}
      <PretrainedShowcase />
    </div>
  )
}
