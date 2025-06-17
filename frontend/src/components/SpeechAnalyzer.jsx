import React, { useState } from 'react';
import { Upload, FileText, BarChart3, Brain, Settings, Zap } from 'lucide-react';

const SpeechAnalyzer = () => {
  const [textContent, setTextContent] = useState('');
  const [useCorefResolution, setUseCorefResolution] = useState(true);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setTextContent(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const analyzeText = async () => {
    if (!textContent.trim()) {
      setError('Please provide text content to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const response = await fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textContent,
          use_coref_resolution: useCorefResolution
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Received data:', data); // Debug log
      setResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Failed to analyze text: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSampleText = () => {
    return `Today I want to talk about artificial intelligence and its impact on society. AI is transforming many industries right now. It's changing how we work and live.

But there are also concerns about AI. Some people worry about job displacement. They think AI might replace human workers. This is a valid concern that we need to address.

However, AI also creates new opportunities. It can help us solve complex problems. For example, it's being used in healthcare to diagnose diseases. AI is also helping in climate research.

The key is to develop AI responsibly. We need proper regulations and ethical guidelines. This will ensure that AI benefits everyone, not just a few companies.

In conclusion, AI is neither good nor bad by itself. It depends on how we use it. We must work together to shape its development for the benefit of humanity.`;
  };

  const loadSampleText = () => {
    setTextContent(getSampleText());
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header Card */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg text-white">
        <div className="p-6">
          <div className="flex items-center gap-3">
            <Brain className="h-8 w-8" />
            <div>
              <h1 className="text-3xl font-bold">Speech Completion Prediction</h1>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Section */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow-md border border-gray-200">
          <div className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="h-5 w-5 text-blue-600" />
              <h2 className="text-xl font-semibold text-gray-900">Text Input</h2>
            </div>
            
            <div className="space-y-4">
              {/* File Upload */}
              <div>
                <label htmlFor="file-upload" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-gray-700 mb-2">
                  <Upload className="h-4 w-4" />
                  Upload Text File (.txt)
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".txt"
                  onChange={handleFileUpload}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Text Area */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="text-content" className="block text-sm font-medium text-gray-700">
                    Or paste/edit text directly:
                  </label>
                  <button
                    onClick={loadSampleText}
                    className="text-sm text-blue-600 hover:text-blue-800 underline"
                  >
                    Load Sample Text
                  </button>
                </div>
                <textarea
                  id="text-content"
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  placeholder="Paste your text here..."
                  className="w-full h-64 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none font-mono text-sm"
                />
                <div className="text-xs text-gray-500 mt-1">
                  Characters: {textContent.length} | Words: {textContent.trim() ? textContent.trim().split(/\s+/).length : 0}
                </div>
              </div>

              {/* AI Processing Options */}
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h3 className="font-medium text-blue-800 mb-3 flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  AI Processing Pipeline
                </h3>
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={useCorefResolution}
                      onChange={(e) => setUseCorefResolution(e.target.checked)}
                      className="mr-2 rounded"
                    />
                    <div>
                      <span className="text-sm font-medium text-blue-800">Coreference Resolution</span>
                      <p className="text-xs text-blue-600">
                        Replace pronouns (he, she, it, they) with their actual referents for better analysis
                      </p>
                    </div>
                  </label>
                  
                  <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                    <strong>Automatic Processing:</strong>
                    <ul className="mt-1 ml-4 list-disc">
                      <li>Smart sentence splitting and merging</li>
                      <li>Removal of very short segments (&lt;3 words)</li>
                      <li>Conjunction-based sentence combining</li>
                      <li>Semantic similarity-based merging</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Advanced Settings (Collapsible)
              <div className="border rounded-lg">
                <button
                  onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                  className="w-full p-3 flex items-center justify-between text-left text-sm hover:bg-gray-50"
                >
                  <div className="flex items-center gap-2">
                    <Settings className="h-4 w-4 text-gray-500" />
                    <span className="font-medium text-gray-700">Advanced Settings</span>
                  </div>
                  <span className="text-gray-400">{showAdvancedSettings ? '−' : '+'}</span> */}
                {/* </button>
                
                {showAdvancedSettings && (
                  <div className="p-4 border-t space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Initial Similarity Threshold: {initialThreshold}
                      </label>
                      <input
                        type="range"
                        min="0.3"
                        max="0.9"
                        step="0.05"
                        value={initialThreshold}
                        onChange={(e) => setInitialThreshold(parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Higher values create fewer, more distinct topic groups
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Minimum Threshold: {minThreshold}
                      </label>
                      <input
                        type="range"
                        min="0.2"
                        max="0.7"
                        step="0.05"
                        value={minThreshold}
                        onChange={(e) => setMinThreshold(parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Minimum similarity threshold as speech progresses
                      </p>
                    </div>
                  </div>
                )}
              </div> */}

              <button 
                onClick={analyzeText} 
                disabled={isAnalyzing || !textContent.trim()}
                className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-md hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4" />
                    Analyze Speech Progress
                  </>
                )}
              </button>

              {error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Info Panel */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2 mb-3">
              <BarChart3 className="h-5 w-5 text-green-600" />
              How It Works
            </h3>
            <div className="space-y-3 text-sm text-gray-600">
              <div>
                <span className="font-medium text-gray-800">1. Preprocessing:</span>
                <p>Resolves pronouns and intelligently splits/merges sentences</p>
              </div>
              <div>
                <span className="font-medium text-gray-800">2. Topic Grouping:</span>
                <p>Uses semantic similarity to group related content</p>
              </div>
              <div>
                <span className="font-medium text-gray-800">3. Progress Tracking:</span>
                <p>Monitors topic staleness and saturation over time</p>
              </div>
              <div>
                <span className="font-medium text-gray-800">4. Dynamic Analysis:</span>
                <p>Adapts thresholds based on content patterns</p>
              </div>
            </div>
          </div>


        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="bg-white rounded-lg shadow-md border border-gray-200">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-green-600" />
              Analysis Results
            </h2>
            
            <div className="space-y-6">
              {/* Progress Summary */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-green-50 p-4 rounded-lg border border-green-200 text-center">
                  <div className="text-2xl font-bold text-green-700">{results.final_progress}</div>
                  <div className="text-sm text-green-600">Final Progress</div>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 text-center">
                  <div className="text-2xl font-bold text-blue-700">{results.total_groups}</div>
                  <div className="text-sm text-blue-600">Topic Groups</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 text-center">
                  <div className="text-2xl font-bold text-purple-700">{results.total_segments}</div>
                  <div className="text-sm text-purple-600">Text Segments</div>
                </div>
              </div>

              {/* Preprocessing Results */}
              {(results.original_sentence_count !== undefined && results.final_sentence_count !== undefined) && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold mb-2 text-gray-800">Preprocessing Results:</h3>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>Original sentences: {results.original_sentence_count}</p>
                    <p>After processing: {results.final_sentence_count}</p>
                    <p>Coreference resolution: {useCorefResolution ? 'Applied' : 'Skipped'}</p>
                    {results.processed_text && (
                      <details className="mt-2">
                        <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                          Show processed text
                        </summary>
                        <div className="mt-2 p-3 bg-white border rounded text-xs font-mono max-h-32 overflow-y-auto">
                          {results.processed_text}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              )}

              {/* Debug Info
              <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                <details>
                  <summary className="cursor-pointer text-sm font-medium text-yellow-800">
                    Debug Information (Click to expand)
                  </summary>
                  <div className="mt-2 text-xs text-yellow-700 font-mono">
                    <pre className="whitespace-pre-wrap max-h-40 overflow-y-auto">
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  </div>
                </details>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-8 border-2 border-dashed border-gray-200 rounded-lg text-center">
                  <BarChart3 className="h-12 w-12 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-500 font-medium">Progress Evolution</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {results.progress_history ? results.progress_history.length : 0} measurement points
                  </p>
                </div>
                <div className="p-8 border-2 border-dashed border-gray-200 rounded-lg text-center">
                  <BarChart3 className="h-12 w-12 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-500 font-medium">Topic Metrics</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Staleness & Saturation patterns
                  </p>
                </div>
              </div> */}

{/* Progress Timeline - Simple Version */}
{results.progress_data && results.progress_data.length > 0 && (
  <div>
    <h3 className="font-semibold mb-3 text-gray-800">Progress Timeline:</h3>
    <div className="overflow-x-auto">
      <table className="w-full border-collapse border border-gray-200 text-sm">
        <thead>
          <tr className="bg-gray-50">
            <th className="border border-gray-200 px-3 py-2 text-left">Position</th>
            <th className="border border-gray-200 px-3 py-2 text-left">Progress</th>
          </tr>
        </thead>
        <tbody>
          {results.progress_data.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              <td className="border border-gray-200 px-3 py-2 font-mono font-semibold text-blue-600">
                {row.position}
              </td>
              <td className="border border-gray-200 px-3 py-2">
                <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-medium">
                  {typeof row.progress === 'number' ? row.progress.toFixed(1) : row.progress}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
)}

              {/* Topic Groups */}
              {results.groups && (
                <div>
                  <h3 className="font-semibold mb-4 text-gray-800">Discovered Topic Groups:</h3>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {results.groups.map((group, i) => (
                      <div key={i} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                        <h4 className="font-medium mb-3 text-blue-700 flex items-center gap-2">
                          <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-xs font-bold">
                            {group.id}
                          </div>
                          Topic Group {group.id} 
                          <span className="text-sm text-gray-500">({group.sentences.length} segments)</span>
                        </h4>
                        <div className="space-y-2">
                          {group.sentences.map((sentence, j) => (
                            <div key={j} className="flex gap-3">
                              <div className="w-1 bg-blue-200 rounded-full flex-shrink-0 mt-1"></div>
                              <p className="text-sm text-gray-700 leading-relaxed">{sentence}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Backend Status */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg">
        <div className="p-6">
          <h2 className="text-lg font-semibold text-blue-800 mb-2 flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            Backend Connection
          </h2>
          <div className="text-sm text-blue-700">
            <p className="mb-1">
              Endpoint: <code className="bg-blue-100 px-2 py-1 rounded font-mono">http://127.0.0.1:5000/api/analyze</code>
            </p>
            <p className="text-blue-600">
              Ensure your Flask backend is running with all AI models loaded (FastCoref, SentenceTransformers, spaCy).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpeechAnalyzer;