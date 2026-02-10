import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import LoadingState from '../components/common/LoadingState'
import { Download, Printer } from 'lucide-react'

export default function PostmortemView() {
  const { incidentId } = useParams()
  const [loading, setLoading] = useState(true)
  const [postmortem, setPostmortem] = useState(null)

  useEffect(() => {
    fetchPostmortem()
  }, [incidentId])

  const fetchPostmortem = async () => {
    try {
      const data = await api.incidents.getPostmortem(incidentId, 'json')
      setPostmortem(data)
    } catch (error) {
      console.error('Failed to fetch postmortem:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadPDF = async () => {
    try {
      const data = await api.incidents.getPostmortem(incidentId, 'pdf')
      // Handle PDF download
      console.log('Download PDF:', data)
    } catch (error) {
      console.error('Failed to download PDF:', error)
    }
  }

  if (loading) {
    return <LoadingState message="Generating postmortem..." />
  }

  if (!postmortem) {
    return <div>Postmortem not available</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Incident Postmortem</h1>
        <div className="flex items-center space-x-2">
          <button onClick={handleDownloadPDF} className="btn btn-secondary flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Download PDF</span>
          </button>
          <button className="btn btn-secondary">
            <Printer className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Postmortem Content */}
      <div className="card max-w-4xl mx-auto">
        <div className="prose prose-sm max-w-none">
          <h2>{postmortem.title}</h2>

          <section>
            <h3>Executive Summary</h3>
            <p>{postmortem.summary}</p>
          </section>

          <section>
            <h3>Root Causes</h3>
            <ul>
              {postmortem.root_causes.map((cause, idx) => (
                <li key={idx}>{cause}</li>
              ))}
            </ul>
          </section>

          <section>
            <h3>Impact Assessment</h3>
            <p>{postmortem.impact}</p>
          </section>

          <section>
            <h3>Resolution</h3>
            <p>{postmortem.resolution}</p>
          </section>

          <section>
            <h3>Lessons Learned</h3>
            <ul>
              {postmortem.lessons_learned.map((lesson, idx) => (
                <li key={idx}>{lesson}</li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </div>
  )
}
