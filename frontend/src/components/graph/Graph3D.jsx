import { Maximize2, Minimize2, Network, RotateCcw, Search, X, ArrowRight, TrendingUp, TrendingDown, ZoomIn } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

// Muted impact colors — easier on the eyes, less luminous
const IMPACT_COLORS = {
  positive: '#2d9d78', // emerald-600
  negative: '#c94c6a', // rose-600
  neutral: '#6366b8',  // indigo-600
}

const IMPACT_GLOW_COLORS = {
  positive: '#047857', // emerald-700
  negative: '#9f1239', // rose-700
  neutral: '#4338ca',  // indigo-700
}

const IMPACT_LINK_COLORS = {
  positive: '#6b9080', // muted teal
  negative: '#9d6b7a', // muted rose
  neutral: '#7c7fb5',  // muted indigo
}

const SELECTED_COLOR = '#d97706' // amber-600 (less harsh)

const METRIC_EXPLANATIONS = {
  margin_proxy: { down: 'Your profit margin is shrinking — you\'re keeping less of each dollar earned. This usually happens when costs rise or refunds increase.', up: 'Your profit margin is healthy — you\'re keeping a good share of revenue after costs.' },
  refund_rate: { down: 'Too many refunds are eating into your revenue. This could signal product quality issues or unmet customer expectations.', up: 'Refund rate is under control.' },
  delivery_delay_rate: { down: 'Deliveries are arriving late too often. Late deliveries frustrate customers and lead to complaints and refunds.', up: 'Deliveries are on time.' },
  churn_proxy: { down: 'More customers are at risk of leaving. This is often caused by poor delivery, bad reviews, or unresolved complaints.', up: 'Customer retention looks good.' },
  ticket_volume: { down: 'Support tickets are spiking. Customers are reaching out more than usual, which may overwhelm your team.', up: 'Support volume is manageable.' },
  ticket_backlog: { down: 'Unresolved support tickets are piling up. Customers waiting too long for help are more likely to leave.', up: 'Support backlog is under control.' },
  fulfillment_backlog: { down: 'Orders are waiting to be fulfilled. A growing backlog delays deliveries and hurts customer satisfaction.', up: 'Fulfillment is keeping up with demand.' },
  supplier_delay_rate: { down: 'Your suppliers are delivering late. This bottleneck cascades into your own delivery delays and backlog.', up: 'Suppliers are delivering on time.' },
  expense_ratio: { down: 'Expenses are too high relative to revenue, squeezing your margins.', up: 'Expense ratio is healthy.' },
  avg_resolution_time: { down: 'It\'s taking too long to resolve support tickets. Slow resolutions frustrate customers and increase churn risk.', up: 'Support issues are being resolved quickly.' },
  ar_aging_amount: { down: 'Unpaid invoices are piling up. Money owed to you is sitting too long, which hurts cash flow.', up: 'Accounts receivable is in good shape.' },
  dso_proxy: { down: 'Customers are taking too long to pay their invoices, which strains your cash position.', up: 'Customers are paying on time.' },
  daily_expenses: { down: 'Daily costs are running high, putting pressure on your bottom line.', up: 'Daily expenses are reasonable.' },
  review_score_avg: { down: 'Customer review scores are dropping. Unhappy customers leave bad reviews, which hurts reputation and future sales.', up: 'Customer reviews are positive.' },
  net_cash_proxy: { down: 'Your cash position is weakening. This can happen when margins shrink or receivables pile up.', up: 'Cash position is strong.' },
}

function explainWhyDragging(node, affectedBy, affects) {
  const id = node?.id || ''
  const specific = METRIC_EXPLANATIONS[id]?.down
  if (specific) return specific

  const name = (node?.name || id).replace(/_/g, ' ')
  const upstreamNames = affectedBy.map((r) => (r.other?.name || '').replace(/_/g, ' ')).filter(Boolean)
  const downstreamNames = affects.map((r) => (r.other?.name || '').replace(/_/g, ' ')).filter(Boolean)

  let explanation = `${name} is performing worse than normal, which is pulling your overall score down.`
  if (upstreamNames.length > 0) {
    explanation += ` This is being caused by problems in ${upstreamNames.join(' and ')}.`
  }
  if (downstreamNames.length > 0) {
    explanation += ` If not addressed, it will also affect ${downstreamNames.join(' and ')}.`
  }
  return explanation
}

function explainWhyHelping(node) {
  const id = node?.id || ''
  const specific = METRIC_EXPLANATIONS[id]?.up
  if (specific) return specific
  const name = (node?.name || id).replace(/_/g, ' ')
  return `${name} is performing well and contributing positively to your business health.`
}

const BG_COLOR = '#0a0f1a' // Slightly softer dark

/**
 * Extracts a stable string ID from a force-graph source/target value.
 * force-graph mutates link.source/target from string IDs into node object
 * refs at runtime, so we must handle both cases.
 */
function stableId(val) {
  if (val == null) return ''
  if (typeof val === 'object') return String(val.id ?? '')
  return String(val)
}

/**
 * Converts Cytoscape graph format to react-force-graph-3d format.
 *
 * react-force-graph-3d **mutates** graphData in-place (replaces link
 * source/target strings with node object refs). To avoid stale refs on
 * re-render we:
 *  1. Always read from the *original* Cytoscape data (.data wrapper).
 *  2. Keep immutable _sourceId / _targetId copies for the side panel.
 *  3. Store __sourceNode / __targetNode refs for connection display.
 */
export function cytoscapeToForceGraph(cytoscapeGraph) {
  if (!cytoscapeGraph?.elements) return { nodes: [], links: [] }

  const cytNodes = cytoscapeGraph.elements.nodes || []
  const cytEdges = cytoscapeGraph.elements.edges || []
  const idToNode = new Map()

  const nodes = cytNodes.map((n) => {
    const data = n.data || n
    const node = {
      id: data.id,
      name: data.label || data.id,
      val: Math.max(2, (data.size || 20) / 4),
      type: data.type,
      impact: data.impact || 'neutral',
      amount: data.amount,
      primary: data.primary,
      ...data,
    }
    idToNode.set(String(data.id), node)
    return node
  })

  const links = cytEdges.map((e) => {
    const data = e.data || e
    const src = stableId(data.source)
    const tgt = stableId(data.target)
    return {
      id: data.id || `edge_${src}_${tgt}`,
      source: src,
      target: tgt,
      _sourceId: src,
      _targetId: tgt,
      label: data.label || 'related',
      impact: data.impact || 'neutral',
      __sourceNode: idToNode.get(src),
      __targetNode: idToNode.get(tgt),
    }
  })

  return { nodes, links }
}

/**
 * Get node IDs connected to the given node (direct neighbors).
 * Uses stable _sourceId/_targetId to survive force-graph mutations.
 */
function getNeighborAndLinkIds(node, links) {
  const neighborIds = new Set()
  const linkIds = new Set()
  const nodeId = node?.id
  if (!nodeId) return { neighborIds, linkIds }

  links.forEach((link) => {
    const srcId = link._sourceId || (typeof link.source === 'object' ? link.source?.id : link.source)
    const tgtId = link._targetId || (typeof link.target === 'object' ? link.target?.id : link.target)
    if (srcId === nodeId || tgtId === nodeId) {
      linkIds.add(link.id)
      if (srcId === nodeId) neighborIds.add(tgtId)
      else neighborIds.add(srcId)
    }
  })
  return { neighborIds, linkIds }
}

/**
 * Create a glowing sphere with inner emissive material.
 */
function createGlowNode(node, color, isSelected, isHighlighted, hasSelection) {
  const size = (node.val || 2) * 1.5
  const opacity = hasSelection ? (isHighlighted ? 1 : 0.15) : 1

  // Main sphere — reduced emissive for softer look
  const geometry = new THREE.SphereGeometry(size, 24, 24)
  const material = new THREE.MeshPhongMaterial({
    color: new THREE.Color(color),
    emissive: new THREE.Color(color),
    emissiveIntensity: isSelected ? 0.35 : 0.15,
    transparent: true,
    opacity,
    shininess: 40,
  })
  const mesh = new THREE.Mesh(geometry, material)

  // Outer glow ring for selected/primary nodes
  if (isSelected || node.primary) {
    const ringGeo = new THREE.RingGeometry(size * 1.3, size * 1.6, 32)
    const ringMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(isSelected ? SELECTED_COLOR : color),
      transparent: true,
      opacity: isSelected ? 0.6 : 0.3,
      side: THREE.DoubleSide,
    })
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.lookAt(0, 0, 1)
    mesh.add(ring)
  }

  // 3D text label
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const label = node.name || node.id
  const fontSize = 36
  ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`
  const textWidth = ctx.measureText(label).width
  canvas.width = Math.min(textWidth + 24, 512)
  canvas.height = fontSize + 16

  // Redraw with sized canvas
  ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  ctx.shadowColor = 'rgba(0,0,0,0.6)'
  ctx.shadowBlur = 4
  const labelColor = hasSelection && !isHighlighted ? 'rgba(100,116,139,0.15)' : '#64748b'
  ctx.fillStyle = labelColor
  ctx.fillText(label, canvas.width / 2, canvas.height / 2)

  const texture = new THREE.CanvasTexture(canvas)
  texture.minFilter = THREE.LinearFilter
  const spriteMat = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: opacity * 0.9,
    depthWrite: false,
  })
  const sprite = new THREE.Sprite(spriteMat)
  sprite.scale.set(canvas.width / 8, canvas.height / 8, 1)
  sprite.position.set(0, size + 4, 0)
  mesh.add(sprite)

  return mesh
}

export default function Graph3D({ graphData, height = 500, width, onNodeClick, title = '3D Graph', legendMode = 'default', loading = false }) {
  const [selectedNode, setSelectedNode] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [searchRaw, setSearchRaw] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const graphRef = useRef(null)
  const containerRef = useRef(null)
  const graphAreaRef = useRef(null)
  const [graphAreaSize, setGraphAreaSize] = useState({ width: null, height: null })
  const bloomAdded = useRef(false)
  const searchInputRef = useRef(null)
  const debounceTimerRef = useRef(null)

  // Convert Cytoscape → force-graph format.  We store the *parsed* result
  // but always feed ForceGraph3D a fresh deep-clone because the library
  // mutates link.source/target in-place (replaces string IDs with node
  // object refs).  Without the clone, tab-switching or re-renders would
  // pass stale object refs from the previous simulation instance.
  const parsedGraph = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] }
    return cytoscapeToForceGraph(graphData)
  }, [graphData])

  // Counter to force ForceGraph3D remount whenever graph content changes.
  const graphKey = useMemo(() => {
    return `fg_${parsedGraph.nodes.length}_${parsedGraph.links.length}_${Date.now()}`
  }, [parsedGraph])

  // Deep-clone: ForceGraph3D mutates link.source/target in-place (replaces
  // strings with node object refs). We must give it fresh objects each time
  // or it will receive stale refs from a destroyed simulation.
  const forceGraphData = useMemo(() => {
    const clonedNodes = parsedGraph.nodes.map((n) => ({ ...n }))
    const nodeById = new Map(clonedNodes.map((n) => [String(n.id), n]))
    const clonedLinks = parsedGraph.links.map((l) => {
      const srcId = l._sourceId || String(l.source?.id ?? l.source ?? '')
      const tgtId = l._targetId || String(l.target?.id ?? l.target ?? '')
      return {
        ...l,
        source: nodeById.get(srcId) ?? srcId,
        target: nodeById.get(tgtId) ?? tgtId,
      }
    })
    return { nodes: clonedNodes, links: clonedLinks }
  }, [parsedGraph])

  const { neighborIds, linkIds } = useMemo(() => {
    if (!selectedNode || !forceGraphData.links.length)
      return { neighborIds: new Set(), linkIds: new Set() }
    return getNeighborAndLinkIds(selectedNode, forceGraphData.links)
  }, [selectedNode, forceGraphData.links])

  const highlightNodeIds = useMemo(() => {
    const s = new Set(neighborIds)
    if (selectedNode) s.add(selectedNode.id)
    return s
  }, [selectedNode, neighborIds])

  // Debounce search input — 300ms
  const handleSearchChange = useCallback((e) => {
    const value = e.target.value
    setSearchRaw(value)
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    debounceTimerRef.current = setTimeout(() => {
      setSearchQuery(value)
    }, 300)
  }, [])

  const handleSearchClear = useCallback(() => {
    setSearchRaw('')
    setSearchQuery('')
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    searchInputRef.current?.focus()
  }, [])

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    }
  }, [])

  // Compute search matches
  const searchMatchIds = useMemo(() => {
    if (!searchQuery.trim()) return null
    const q = searchQuery.trim().toLowerCase()
    const matched = new Set()
    forceGraphData.nodes.forEach((node) => {
      const name = (node.name || '').toLowerCase()
      const id = (node.id || '').toLowerCase()
      if (name.includes(q) || id.includes(q)) matched.add(node.id)
    })
    return matched
  }, [searchQuery, forceGraphData.nodes])

  // Auto-focus camera when exactly one search match
  useEffect(() => {
    if (!searchMatchIds || searchMatchIds.size !== 1) return
    const matchedId = [...searchMatchIds][0]
    const matchedNode = forceGraphData.nodes.find((n) => n.id === matchedId)
    if (!matchedNode) return
    const fg = graphRef.current
    if (!fg) return
    // Small delay to let the graph settle
    const timer = setTimeout(() => {
      const node = matchedNode
      const distance = 120
      const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0)
      fg.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
        { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
        1200
      )
    }, 150)
    return () => clearTimeout(timer)
  }, [searchMatchIds, forceGraphData.nodes])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Escape: clear selected node AND clear search
      if (e.key === 'Escape') {
        setSelectedNode(null)
        setSearchRaw('')
        setSearchQuery('')
        if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
        return
      }
      // Ctrl/Cmd+F: focus search input
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault()
        searchInputRef.current?.focus()
        searchInputRef.current?.select()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Measure graph area for explicit canvas dimensions (ensures proper centering)
  useLayoutEffect(() => {
    const el = graphAreaRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect
        setGraphAreaSize({ width: Math.round(width), height: Math.round(height) })
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Add bloom post-processing once the graph engine is ready
  useEffect(() => {
    const fg = graphRef.current
    if (!fg || bloomAdded.current) return

    const renderer = fg.renderer()
    const scene = fg.scene()
    if (!renderer || !scene) return

    try {
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.35,  // strength — reduced to avoid harsh glow
        0.3,   // radius
        0.5    // threshold — only bright elements bloom
      )
      const composer = fg.postProcessingComposer()
      composer.addPass(bloomPass)
      bloomAdded.current = true

      // Softer lighting — less intense
      const ambientLight = new THREE.AmbientLight(0x304050, 0.5)
      scene.add(ambientLight)
      const pointLight = new THREE.PointLight(0x6366f1, 0.5, 500)
      pointLight.position.set(50, 100, 50)
      scene.add(pointLight)
      const pointLight2 = new THREE.PointLight(0x10b981, 0.35, 400)
      pointLight2.position.set(-50, -50, 80)
      scene.add(pointLight2)
    } catch {
      // Bloom not supported in this environment
    }
  }, [forceGraphData])

  // Zoom to fit + spread nodes (bigger graphs = more spread, less clustering)
  // Re-run when graph data or container size changes (ensures centered view)
  useEffect(() => {
    const fg = graphRef.current
    if (!fg || !forceGraphData.nodes.length) return
    const n = forceGraphData.nodes.length
    const charge = fg.d3Force?.('charge')
    const link = fg.d3Force?.('link')
    if (charge) {
      charge.strength(-(25 + Math.min(n * 3, 150)))
      charge.distanceMax(100 + Math.min(n * 5, 300))
    }
    if (link) {
      link.distance(60 + Math.min(n * 4, 200))
    }
    const timer = setTimeout(() => {
      fg.zoomToFit(600, 80)
    }, 800)
    return () => clearTimeout(timer)
  }, [forceGraphData, graphAreaSize.width, graphAreaSize.height])

  const handleNodeClick = useCallback(
    (node) => {
      setSelectedNode((prev) => (prev?.id === node?.id ? null : node))
      onNodeClick?.(node)

      // Focus camera on clicked node
      const fg = graphRef.current
      if (fg && node) {
        const distance = 120
        const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0)
        fg.cameraPosition(
          { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
          { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
          1200
        )
      }
    },
    [onNodeClick]
  )

  const handleZoomToFit = useCallback(() => {
    graphRef.current?.zoomToFit(600, 60)
  }, [])

  const handleResetSelection = useCallback(() => {
    setSelectedNode(null)
    graphRef.current?.zoomToFit(600, 60)
  }, [])

  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().catch(() => {})
      setIsFullscreen(true)
    } else {
      document.exitFullscreen?.().catch(() => {})
      setIsFullscreen(false)
    }
  }, [])

  // Listen for fullscreen exit via Escape
  useEffect(() => {
    const handler = () => {
      if (!document.fullscreenElement) setIsFullscreen(false)
    }
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  const nodeThreeObject = useCallback(
    (node) => {
      const isSelected = selectedNode?.id === node?.id
      const isNeighborHighlighted = highlightNodeIds.has(node?.id)
      const hasSelection = !!selectedNode
      const isSearchActive = searchMatchIds !== null
      const isSearchMatch = isSearchActive ? searchMatchIds.has(node?.id) : false

      // Combine selection highlight and search match for the "is highlighted" concept
      const isHighlighted = hasSelection ? isNeighborHighlighted : (isSearchActive ? isSearchMatch : true)
      // Something dims a node if: there is a selection and it's not a neighbor/selected,
      // OR there is a search and it does not match
      const isDimmed = (hasSelection && !isNeighborHighlighted && !isSelected) || (isSearchActive && !isSearchMatch)

      let color
      if (isSelected) {
        color = SELECTED_COLOR
      } else if (isDimmed) {
        color = '#1e293b' // slate-800
      } else {
        color = IMPACT_COLORS[node?.impact] || IMPACT_COLORS.neutral
      }

      const effectiveHasSelection = hasSelection || isSearchActive
      const effectiveIsHighlighted = isSelected || (hasSelection ? isNeighborHighlighted : true) && (!isSearchActive || isSearchMatch)

      return createGlowNode(node, color, isSelected, effectiveIsHighlighted, effectiveHasSelection && isDimmed ? true : hasSelection)
    },
    [selectedNode, highlightNodeIds, searchMatchIds]
  )

  const getLinkColor = useCallback(
    (link) => {
      if (!selectedNode) {
        return IMPACT_LINK_COLORS[link?.impact] || IMPACT_LINK_COLORS.neutral
      }
      return linkIds.has(link?.id)
        ? IMPACT_LINK_COLORS[link?.impact] || IMPACT_LINK_COLORS.neutral
        : 'rgba(30,41,59,0.3)'
    },
    [selectedNode, linkIds]
  )

  const getLinkWidth = useCallback(
    (link) => {
      if (!selectedNode) return 1.5
      return linkIds.has(link?.id) ? 3 : 0.3
    },
    [selectedNode, linkIds]
  )

  const getLinkParticles = useCallback(
    (link) => {
      if (!selectedNode) return 2
      return linkIds.has(link?.id) ? 4 : 0
    },
    [selectedNode, linkIds]
  )

  const getLinkParticleWidth = useCallback(
    (link) => {
      return linkIds.has(link?.id) ? 2.5 : 1.2
    },
    [linkIds]
  )

  const { affectedBy, affects } = useMemo(() => {
    if (!selectedNode || !forceGraphData.links.length) return { affectedBy: [], affects: [] }
    const incoming = []
    const outgoing = []
    forceGraphData.links.forEach((link) => {
      const srcId = link._sourceId || (typeof link.source === 'object' ? link.source?.id : link.source)
      const tgtId = link._targetId || (typeof link.target === 'object' ? link.target?.id : link.target)
      const srcNode = link.__sourceNode || (typeof link.source === 'object' ? link.source : null)
      const tgtNode = link.__targetNode || (typeof link.target === 'object' ? link.target : null)
      if (srcId === selectedNode.id) {
        outgoing.push({ other: tgtNode || { id: tgtId, name: tgtId }, label: link.label, impact: link.impact })
      } else if (tgtId === selectedNode.id) {
        incoming.push({ other: srcNode || { id: srcId, name: srcId }, label: link.label, impact: link.impact })
      }
    })
    return { affectedBy: incoming, affects: outgoing }
  }, [selectedNode, forceGraphData.links])

  // Loading state — shown when loading prop is true OR graphData is null/undefined
  if (loading || graphData === null || graphData === undefined) {
    return (
      <div
        className="rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center"
        style={{ minHeight: height }}
      >
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-12 h-12">
            <div className="absolute inset-0 rounded-full border-2 border-slate-700" />
            <div className="absolute inset-0 rounded-full border-2 border-t-slate-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
          </div>
          <div className="text-center">
            <p className="text-slate-400 text-sm font-medium">Loading graph</p>
            <p className="text-slate-600 text-xs mt-0.5">Building causal relationships</p>
          </div>
        </div>
      </div>
    )
  }

  if (!forceGraphData.nodes.length) {
    return (
      <div
        className="rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 to-slate-800 p-12 text-center"
        style={{ minHeight: height }}
      >
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-700/50 flex items-center justify-center">
          <Network className="w-8 h-8 text-slate-400" />
        </div>
        <p className="text-slate-400 font-medium">No graph data available</p>
        <p className="text-slate-500 text-sm mt-1">Run analysis to generate causal relationships</p>
      </div>
    )
  }

  const graphHeight = isFullscreen ? '100vh' : height
  const totalNodes = forceGraphData.nodes.length
  const matchCount = searchMatchIds ? searchMatchIds.size : 0

  return (
    <div ref={containerRef} className="flex gap-0 w-full max-w-full bg-[#080c18] rounded-2xl overflow-hidden border border-slate-700/50 shadow-xl" style={{ height: graphHeight, minWidth: 0 }}>
      {/* Graph area */}
      <div ref={graphAreaRef} className="flex-1 relative min-w-[200px] min-h-0">
        {/* Header bar */}
        <div className="absolute top-0 left-0 right-0 z-10 px-4 py-2.5 bg-gradient-to-b from-[#080c18] via-[#080c18]/90 to-transparent flex items-center justify-between pointer-events-none">
          <div className="pointer-events-auto flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10">
              <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-pulse" />
              <span className="text-xs font-medium text-slate-500 tracking-wide">{title}</span>
            </div>
            <span className="text-[10px] text-slate-600 hidden sm:inline">
              Drag to rotate &middot; Scroll to zoom &middot; Click node to explore
            </span>
          </div>
          <div className="pointer-events-auto flex items-center gap-1.5">
            {/* Search input */}
            <div className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/5 backdrop-blur-md border border-white/10 focus-within:border-slate-500/60 transition-colors">
              <Search className="w-3 h-3 text-slate-500 flex-shrink-0" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchRaw}
                onChange={handleSearchChange}
                placeholder="Search nodes..."
                className="bg-transparent text-xs text-slate-300 placeholder-slate-600 outline-none w-28 min-w-0"
                aria-label="Search graph nodes"
              />
              {searchRaw && (
                <>
                  <span className="text-[10px] text-slate-500 font-medium whitespace-nowrap px-1">
                    {matchCount}/{totalNodes}
                  </span>
                  <button
                    type="button"
                    onClick={handleSearchClear}
                    className="text-slate-500 hover:text-slate-300 transition-colors flex-shrink-0"
                    aria-label="Clear search"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </>
              )}
            </div>

            {selectedNode && (
              <button
                type="button"
                onClick={handleResetSelection}
                className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-400 hover:text-white transition-colors"
                title="Reset view"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}
            <button
              type="button"
              onClick={handleZoomToFit}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-400 hover:text-white transition-colors"
              title="Zoom to fit"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-400 hover:text-white transition-colors"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-3 px-3 py-1.5 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
          <span className="flex items-center gap-1.5 text-[10px] text-slate-500">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: IMPACT_COLORS.positive }} />
            {legendMode === 'forecast' ? 'Driving up' : 'Profit'}
          </span>
          <span className="flex items-center gap-1.5 text-[10px] text-slate-500">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: IMPACT_COLORS.negative }} />
            {legendMode === 'forecast' ? 'Driving down' : 'Loss'}
          </span>
          <span className="flex items-center gap-1.5 text-[10px] text-slate-500">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: IMPACT_COLORS.neutral }} />
            Neutral
          </span>
        </div>

        {/* Node count badge */}
        <div className="absolute bottom-3 right-3 z-10 px-2.5 py-1 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
          <span className="text-[10px] text-slate-500 font-medium">
            {forceGraphData.nodes.length} nodes &middot; {forceGraphData.links.length} links
          </span>
        </div>

        <ForceGraph3D
          key={graphKey}
          ref={graphRef}
          graphData={forceGraphData}
          width={width ?? graphAreaSize.width ?? undefined}
          height={typeof graphHeight === 'number' ? graphHeight : (graphAreaSize.height ?? undefined)}
          backgroundColor={BG_COLOR}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          linkColor={getLinkColor}
          linkWidth={getLinkWidth}
          linkOpacity={0.8}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={0.9}
          linkDirectionalArrowColor={getLinkColor}
          linkDirectionalParticles={getLinkParticles}
          linkDirectionalParticleWidth={getLinkParticleWidth}
          linkDirectionalParticleSpeed={0.004}
          linkDirectionalParticleColor={getLinkColor}
          linkCurvature={0.1}
          onNodeClick={handleNodeClick}
          onBackgroundClick={() => setSelectedNode(null)}
          enableNodeDrag={true}
          warmupTicks={50}
          cooldownTicks={120}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.2}
        />
      </div>

      {/* Side panel — always visible; shows connections when node selected */}
      <div className="w-72 min-w-[260px] sm:w-80 sm:min-w-[280px] flex-shrink-0 border-l border-slate-700/50 bg-gradient-to-b from-slate-900 to-[#0c1024] overflow-y-auto flex flex-col">
        <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
          <h3 className="font-semibold text-slate-200 text-sm">Connections</h3>
          {selectedNode && (
            <button
              type="button"
              onClick={() => setSelectedNode(null)}
              className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
              aria-label="Close"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <div className="p-4 space-y-4 flex-1">
          {!selectedNode ? (
            <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
              <div className="w-12 h-12 rounded-xl bg-slate-700/50 flex items-center justify-center mb-3">
                <Network className="w-6 h-6 text-slate-500" />
              </div>
              <p className="text-sm text-slate-400 font-medium">Click a node</p>
              <p className="text-xs text-slate-600 mt-1">to see what it connects from and to</p>
            </div>
          ) : (
            <>
            {/* Connection flow diagram — arrows + node names */}
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-600/40">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-3">Connection flow</p>
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                {affectedBy.length === 0 ? (
                  <span className="text-slate-500 italic">—</span>
                ) : (
                  affectedBy.map((r, i) => {
                    const label = (r.other?.name || r.other?.id || '?').replace(/_/g, ' ')
                    return (
                      <span key={`flow-in-${i}`} className="flex items-center gap-1">
                        <span
                          className="px-1.5 py-0.5 rounded"
                          style={{
                            backgroundColor: `${IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral}30`,
                            color: IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral,
                          }}
                        >
                          {label}
                        </span>
                        {i < affectedBy.length - 1 && <span className="text-slate-500">+</span>}
                      </span>
                    )
                  })
                )}
                {(affectedBy.length > 0 || affects.length > 0) && (
                  <ArrowRight className="w-4 h-4 text-slate-500 flex-shrink-0 mx-0.5" />
                )}
                <span
                  className="px-2 py-1 rounded font-semibold"
                  style={{
                    backgroundColor: `${IMPACT_COLORS[selectedNode.impact] || IMPACT_COLORS.neutral}40`,
                    color: IMPACT_COLORS[selectedNode.impact] || IMPACT_COLORS.neutral,
                  }}
                >
                  {(selectedNode.name || selectedNode.id).replace(/_/g, ' ')}
                </span>
                {affects.length > 0 && (
                  <>
                    <ArrowRight className="w-4 h-4 text-slate-500 flex-shrink-0 mx-0.5" />
                    {affects.map((r, i) => {
                      const label = (r.other?.name || r.other?.id || '?').replace(/_/g, ' ')
                      return (
                        <span key={`flow-out-${i}`} className="flex items-center gap-1">
                          <span
                            className="px-1.5 py-0.5 rounded"
                            style={{
                              backgroundColor: `${IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral}30`,
                              color: IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral,
                            }}
                          >
                            {label}
                          </span>
                          {i < affects.length - 1 && <span className="text-slate-500">+</span>}
                        </span>
                      )
                    })}
                  </>
                )}
              </div>
              <p className="text-[10px] text-slate-600 mt-2">
                {affectedBy.length > 0 && `Connects from ${affectedBy.length} node(s)`}
                {affectedBy.length > 0 && affects.length > 0 && ' • '}
                {affects.length > 0 && `Connects to ${affects.length} node(s)`}
              </p>
            </div>

            {/* Negative node callout — explains WHY it drags the score down */}
            {selectedNode.impact === 'negative' && (
              <div className="p-3 rounded-xl border border-rose-500/40 bg-rose-950/30">
                <p className="text-xs font-semibold text-rose-300 mb-1">Driving score down</p>
                <p className="text-[11px] text-rose-200/90">
                  {explainWhyDragging(selectedNode, affectedBy, affects)}
                </p>
                {(affectedBy.length > 0 || affects.length > 0) && (
                  <div className="mt-2 text-[11px] text-rose-200/80">
                    {affectedBy.length > 0 && (
                      <span>Caused by: {affectedBy.map((r) => (r.other?.name || r.other?.id || '?').replace(/_/g, ' ')).join(', ')}</span>
                    )}
                    {affectedBy.length > 0 && affects.length > 0 && ' → '}
                    {affects.length > 0 && (
                      <span>Leading to: {affects.map((r) => (r.other?.name || r.other?.id || '?').replace(/_/g, ' ')).join(', ')}</span>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Positive node callout — explains what's going well */}
            {selectedNode.impact === 'positive' && (
              <div className="p-3 rounded-xl border border-emerald-500/40 bg-emerald-950/30">
                <p className="text-xs font-semibold text-emerald-300 mb-1">Helping your score</p>
                <p className="text-[11px] text-emerald-200/90">
                  {explainWhyHelping(selectedNode)}
                </p>
              </div>
            )}

            {/* Detailed lists: Connects from | Connects to */}
            <div className="space-y-3">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                  Connects from ({affectedBy.length})
                </p>
                <div className="space-y-1.5 max-h-28 overflow-y-auto pr-1">
                  {affectedBy.length === 0 ? (
                    <p className="text-xs text-slate-600 italic">Nothing upstream</p>
                  ) : (
                    affectedBy.map((r, i) => (
                      <div key={`in-${i}`} className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/5">
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ backgroundColor: IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral }}
                        />
                        <span className="text-slate-300 font-medium text-xs">
                          {(r.other?.name || r.other?.id || '?').replace(/_/g, ' ')}
                        </span>
                        <ArrowRight className="w-3 h-3 text-slate-500 flex-shrink-0 ml-auto" />
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-2 flex items-center gap-2">
                  <ArrowRight className="w-3 h-3 text-slate-500" />
                  Connects to ({affects.length})
                </p>
                <div className="space-y-1.5 max-h-28 overflow-y-auto pr-1">
                  {affects.length === 0 ? (
                    <p className="text-xs text-slate-600 italic">Nothing downstream</p>
                  ) : (
                    affects.map((r, i) => (
                      <div key={`out-${i}`} className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/5">
                        <ArrowRight className="w-3 h-3 text-slate-500 flex-shrink-0" />
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ backgroundColor: IMPACT_COLORS[r.impact] || IMPACT_COLORS.neutral }}
                        />
                        <span className="text-slate-300 font-medium text-xs">
                          {(r.other?.name || r.other?.id || '?').replace(/_/g, ' ')}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
