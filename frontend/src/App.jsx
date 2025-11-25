import { useState, useCallback } from 'react';
import ChartComponent from './ChartComponent';
import HeatmapComponent from './HeatmapComponent';

function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [imageName, setImageName] = useState('');
  const [userLabel, setUserLabel] = useState('');

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (uploadedFile) => {
    if (!uploadedFile.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }
    setFile(uploadedFile);
    setError(null);
    setResult(null);
    
    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(uploadedFile);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setProgress(0);
    setProgressText('Uploading image...');

    const formData = new FormData();
    formData.append('file', file);
    
    // Add image name and user label if provided
    if (imageName) {
      formData.append('image_name', imageName);
    }
    if (userLabel) {
      formData.append('user_label', userLabel);
    }

    try {
      // Simulate progress stages
      const progressSteps = [
        { progress: 15, text: 'Uploading image...', delay: 300 },
        { progress: 30, text: 'Extracting noise residuals...', delay: 500 },
        { progress: 50, text: 'Analyzing GGD parameters...', delay: 800 },
        { progress: 65, text: 'Computing gradient statistics...', delay: 600 },
        { progress: 80, text: 'Analyzing local patches...', delay: 700 },
        { progress: 95, text: 'Finalizing detection...', delay: 400 },
      ];

      let currentStep = 0;
      const progressInterval = setInterval(() => {
        if (currentStep < progressSteps.length) {
          const step = progressSteps[currentStep];
          setProgress(step.progress);
          setProgressText(step.text);
          currentStep++;
        }
      }, 400);

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setProgress(100);
      setProgressText('Analysis complete!');
      
      setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, 500);
      
    } catch (err) {
      setError(err.message || 'Failed to analyze image');
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    setImageName('');
    setUserLabel('');
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-cyan-500/30 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
            AI Image Forensics
          </h1>
          <p className="text-gray-400 mt-2">
            Generalized Gaussian Distribution Analysis for AI Detection
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        {/* Upload Section */}
        <div className="max-w-4xl mx-auto mb-12">
          <div
            className={`
              relative border-2 border-dashed rounded-lg p-12 text-center transition-all
              ${dragActive 
                ? 'border-cyan-400 bg-cyan-500/10' 
                : 'border-cyan-500/30 bg-gray-900/50'
              }
              ${file ? 'border-cyan-400' : ''}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              accept="image/*"
              onChange={handleChange}
              className="hidden"
            />
            
            {!file ? (
              <label htmlFor="file-upload" className="cursor-pointer">
                <div className="mb-4">
                  <svg
                    className="mx-auto h-16 w-16 text-cyan-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                    aria-hidden="true"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
                <p className="text-xl text-cyan-400 font-semibold mb-2">
                  Drop your image here or click to browse
                </p>
                <p className="text-gray-400 text-sm">
                  Supports JPG, PNG, WebP and other common formats
                </p>
              </label>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-center gap-3">
                  <svg
                    className="h-8 w-8 text-green-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <p className="text-xl text-cyan-400 font-semibold">
                    {file.name}
                  </p>
                </div>
                <p className="text-gray-400 text-sm">
                  {(file.size / 1024).toFixed(2)} KB
                </p>
                
                {/* Experiment Input Fields */}
                <div className="mt-6 space-y-3 max-w-md mx-auto">
                  <div>
                    <label className="block text-sm font-medium text-cyan-400 mb-1">
                      Image Name (for database)
                    </label>
                    <input
                      type="text"
                      value={imageName}
                      onChange={(e) => setImageName(e.target.value)}
                      placeholder="e.g., test_image_001"
                      className="w-full px-4 py-2 bg-gray-800 border border-cyan-500/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-cyan-400 mb-1">
                      Your Label (ground truth)
                    </label>
                    <select
                      value={userLabel}
                      onChange={(e) => setUserLabel(e.target.value)}
                      className="w-full px-4 py-2 bg-gray-800 border border-cyan-500/30 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    >
                      <option value="">Select label...</option>
                      <option value="AI">AI Generated</option>
                      <option value="Real">Real Photo</option>
                      <option value="Smartphone">Smartphone Photo</option>
                      <option value="DSLR">DSLR Photo</option>
                      <option value="Edited">Edited Photo</option>
                      <option value="Unknown">Unknown</option>
                    </select>
                  </div>
                  {imageName && userLabel && (
                    <p className="text-xs text-green-400 text-center">
                      ✓ This experiment will be saved to database
                    </p>
                  )}
                </div>
                
                <div className="flex gap-4 justify-center mt-6">
                  <button
                    onClick={analyzeImage}
                    disabled={loading}
                    className="
                      px-8 py-3 bg-cyan-500 hover:bg-cyan-600 
                      text-gray-900 font-bold rounded-lg
                      transition-all transform hover:scale-105
                      disabled:opacity-50 disabled:cursor-not-allowed
                      disabled:hover:scale-100
                      shadow-lg shadow-cyan-500/50
                    "
                  >
                    {loading ? 'Analyzing...' : 'Analyze Image'}
                  </button>
                  
                  <button
                    onClick={reset}
                    disabled={loading}
                    className="
                      !px-8 !py-3 !bg-gray-700 hover:bg-gray-600 
                      !text-gray font-bold rounded-lg
                      transition-all
                      disabled:opacity-50 disabled:cursor-not-allowed
                    "
                  >
                    Reset 
                  </button>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500 rounded-lg">
              <p className="text-red-400 text-center font-semibold">
                {error}
              </p>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="max-w-4xl mx-auto mb-12">
            <div className="bg-gray-900/50 border border-cyan-500/30 rounded-lg p-12">
              <div className="flex flex-col items-center gap-6">
                <div className="relative w-20 h-20">
                  <div className="absolute inset-0 border-4 border-cyan-500/30 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-cyan-500 rounded-full border-t-transparent animate-spin"></div>
                </div>
                
                {/* Progress Text */}
                <div className="text-center">
                  <p className="text-xl text-cyan-400 font-semibold">
                    {progressText}
                  </p>
                  <p className="text-gray-400 text-sm mt-2">
                    Performing forensic analysis
                  </p>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full max-w-md">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-400">Progress</span>
                    <span className="text-sm font-bold text-cyan-400">{progress}%</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300 ease-out rounded-full"
                      style={{ width: `${progress}%` }}
                    >
                      <div className="h-full w-full animate-pulse bg-white/20"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {result && !loading && (
          <div className="max-w-7xl mx-auto">
            {/* Detection Verdict */}
            <div className={`mb-8 p-6 rounded-lg border-2 ${
              result.detection.verdict === 'uncertain'
                ? 'bg-yellow-500/10 border-yellow-500'
                : result.detection.is_ai_generated
                ? 'bg-red-500/10 border-red-500'
                : 'bg-green-500/10 border-green-500'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {result.detection.verdict === 'uncertain' ? (
                    <svg className="w-12 h-12 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : result.detection.is_ai_generated ? (
                    <svg className="w-12 h-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  ) : (
                    <svg className="w-12 h-12 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                  <div>
                    <h2 className={`text-2xl font-bold ${
                      result.detection.verdict === 'uncertain'
                        ? 'text-yellow-400'
                        : result.detection.is_ai_generated ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {result.detection.verdict === 'uncertain'
                        ? 'Uncertain - Potential AI/Alteration'
                        : result.detection.is_ai_generated ? 'AI Generated' : 'Likely Real'}
                    </h2>
                    <p className="text-gray-400 text-sm mt-1">
                      {result.detection.reason}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-4xl font-bold ${
                    result.detection.verdict === 'uncertain'
                      ? 'text-yellow-400'
                      : result.detection.is_ai_generated ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {result.detection.confidence}%
                  </div>
                  <p className="text-gray-400 text-xs mt-1">Confidence</p>
                </div>
              </div>
            </div>

            {/* Main Content: 2 Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Left Column: Image */}
              <div className="bg-gray-900 rounded-lg border border-cyan-500/30 p-6">
                <h3 className="text-lg font-bold text-cyan-400 mb-4">
                  Analyzed Image
                </h3>
                <div className="relative bg-gray-950 rounded-lg overflow-hidden" style={{aspectRatio: '1/1'}}>
                  {imagePreview && (
                    <img 
                      src={imagePreview} 
                      alt="Uploaded" 
                      className="w-full h-full object-contain"
                    />
                  )}
                </div>
                <div className="mt-3 text-center text-gray-400 text-sm truncate">
                  {file?.name}
                </div>
              </div>

              {/* Right Column: Visualizations */}
              <div className="space-y-6">
                {/* Heatmap */}
                <HeatmapComponent data={result} />
                
                {/* Chart */}
                <ChartComponent data={result} />
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              <div className="bg-gray-900 border border-cyan-500/30 rounded-lg p-4">
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  Noise β
                </h3>
                <p className="text-gray-300 text-xl font-bold">
                  {result.detection.metrics.ggd_shape.toFixed(3)}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.ggd_shape > 1.8 
                    ? 'AI-like' 
                    : result.detection.metrics.ggd_shape > 1.5 
                    ? 'Elevated'
                    : 'Natural'}
                </p>
              </div>
              
              <div className={`bg-gray-900 border rounded-lg p-4 ${
                result.detection.metrics.texture_inconsistency > 0.5 
                  ? 'border-red-500/50' 
                  : result.detection.metrics.texture_inconsistency > 0.3 
                  ? 'border-yellow-500/50'
                  : 'border-cyan-500/30'
              }`}>
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  Texture Gap
                </h3>
                <p className={`text-xl font-bold ${
                  result.detection.metrics.texture_inconsistency > 0.5 
                    ? 'text-red-400' 
                    : result.detection.metrics.texture_inconsistency > 0.3 
                    ? 'text-yellow-400'
                    : 'text-gray-300'
                }`}>
                  {result.detection.metrics.texture_inconsistency.toFixed(3)}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.texture_inconsistency > 0.5 
                    ? 'Inconsistent' 
                    : result.detection.metrics.texture_inconsistency > 0.3 
                    ? 'Moderate'
                    : 'Consistent'}
                </p>
              </div>
              
              <div className={`bg-gray-900 border rounded-lg p-4 ${
                result.detection.metrics.local_suspicious_ratio > 0.4 
                  ? 'border-red-500/50' 
                  : result.detection.metrics.local_suspicious_ratio > 0.25 
                  ? 'border-yellow-500/50'
                  : 'border-cyan-500/30'
              }`}>
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  Suspicious Patches
                </h3>
                <p className={`text-xl font-bold ${
                  result.detection.metrics.local_suspicious_ratio > 0.4 
                    ? 'text-red-400' 
                    : result.detection.metrics.local_suspicious_ratio > 0.25 
                    ? 'text-yellow-400'
                    : 'text-gray-300'
                }`}>
                  {result.detection.metrics.local_suspicious_patches}/{result.detection.metrics.blocks_analyzed}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {(result.detection.metrics.local_suspicious_ratio * 100).toFixed(0)}% suspicious
                </p>
              </div>
              
              <div className="bg-gray-900 border border-cyan-500/30 rounded-lg p-4">
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  Patch Variance
                </h3>
                <p className="text-gray-300 text-xl font-bold">
                  {result.detection.metrics.local_variance.toFixed(3)}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.local_variance > 0.3 
                    ? 'Inconsistent' 
                    : 'Uniform'}
                </p>
              </div>
              
              {result.detection.metrics.ssp_found && (
                <div className={`bg-gray-900 border rounded-lg p-4 ${
                  (result.detection.metrics.ssp_beta >= 1.7 && result.detection.metrics.ssp_beta <= 2.3) || result.detection.metrics.ssp_beta > 5.0
                    ? 'border-red-500/50' 
                    : result.detection.metrics.ssp_beta < 1.0
                    ? 'border-green-500/50'
                    : 'border-cyan-500/30'
                }`}>
                  <h3 className="text-sm font-bold text-cyan-400 mb-2">
                    SSP β {result.detection.metrics.ssp_used_tiebreaker && '⚖️'}
                  </h3>
                  <p className={`text-xl font-bold ${
                    (result.detection.metrics.ssp_beta >= 1.7 && result.detection.metrics.ssp_beta <= 2.3) || result.detection.metrics.ssp_beta > 5.0
                      ? 'text-red-400' 
                      : result.detection.metrics.ssp_beta < 1.0
                      ? 'text-green-400'
                      : 'text-gray-300'
                  }`}>
                    {result.detection.metrics.ssp_beta.toFixed(3)}
                  </p>
                  <p className="text-gray-500 text-xs mt-1">
                    {result.detection.metrics.ssp_beta >= 1.7 && result.detection.metrics.ssp_beta <= 2.3
                      ? 'Gaussian-like' 
                      : result.detection.metrics.ssp_beta > 5.0
                      ? 'Over-smoothed'
                      : result.detection.metrics.ssp_beta < 1.0
                      ? 'Sensor noise'
                      : 'Moderate'}
                  </p>
                </div>
              )}
              
              <div className={`bg-gray-900 border rounded-lg p-4 ${
                result.detection.metrics.patchcraft_score > 0.7 
                  ? 'border-red-500/50' 
                  : result.detection.metrics.patchcraft_score > 0.5 
                  ? 'border-yellow-500/50'
                  : 'border-cyan-500/30'
              }`}>
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  PatchCraft SRM
                </h3>
                <p className={`text-xl font-bold ${
                  result.detection.metrics.patchcraft_score > 0.7 
                    ? 'text-red-400' 
                    : result.detection.metrics.patchcraft_score > 0.5 
                    ? 'text-yellow-400'
                    : 'text-gray-300'
                }`}>
                  {result.detection.metrics.patchcraft_score.toFixed(3)}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.patchcraft_score > 0.7 
                    ? 'Suspicious' 
                    : result.detection.metrics.patchcraft_score > 0.5 
                    ? 'Moderate'
                    : 'Natural'}
                </p>
              </div>

              <div className={`bg-gray-900 border rounded-lg p-4 ${
                result.detection.metrics.spectral_score > 0.6 
                  ? 'border-red-500/50' 
                  : result.detection.metrics.spectral_score > 0.4 
                  ? 'border-yellow-500/50'
                  : 'border-cyan-500/30'
              }`}>
                <h3 className="text-sm font-bold text-cyan-400 mb-2">
                  Spectral Grid
                </h3>
                <p className={`text-xl font-bold ${
                  result.detection.metrics.spectral_score > 0.6 
                    ? 'text-red-400' 
                    : result.detection.metrics.spectral_score > 0.4 
                    ? 'text-yellow-400'
                    : 'text-gray-300'
                }`}>
                  {result.detection.metrics.spectral_score.toFixed(3)}
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.spectral_score > 0.6 
                    ? 'Grid artifacts' 
                    : result.detection.metrics.spectral_score > 0.4 
                    ? 'Weak signal'
                    : 'No artifacts'}
                </p>
              </div>
              
              {result.detection.metrics.lca_found && (
                <div className={`bg-gray-900 border rounded-lg p-4 ${
                  result.detection.metrics.lca_displacement < 0.1 
                    ? 'border-red-500/50' 
                    : result.detection.metrics.lca_displacement < 0.5
                    ? 'border-yellow-500/50'
                    : result.detection.metrics.lca_displacement > 1.0
                    ? 'border-green-500/50'
                    : 'border-cyan-500/30'
                }`}>
                  <h3 className="text-sm font-bold text-cyan-400 mb-2">
                    Lens Aberration
                  </h3>
                  <p className={`text-xl font-bold ${
                    result.detection.metrics.lca_displacement < 0.1 
                      ? 'text-red-400' 
                      : result.detection.metrics.lca_displacement < 0.5
                      ? 'text-yellow-400'
                      : result.detection.metrics.lca_displacement > 1.0
                      ? 'text-green-400'
                      : 'text-gray-300'
                  }`}>
                    {result.detection.metrics.lca_displacement.toFixed(2)}%
                  </p>
                  <p className="text-gray-500 text-xs mt-1">
                    {result.detection.metrics.lca_displacement < 0.1 
                      ? 'No CA pattern' 
                      : result.detection.metrics.lca_displacement < 0.5
                      ? 'Weak pattern'
                      : result.detection.metrics.lca_displacement > 1.0
                      ? 'Natural lens'
                      : 'Moderate'}
                  </p>
                </div>
              )}
              
              <div className="bg-gray-900 border border-red-500/30 rounded-lg p-4">
                <h3 className="text-sm font-bold text-red-400 mb-2">
                  AI Score
                </h3>
                <p className="text-gray-300 text-xl font-bold">
                  {result.detection.metrics.ai_score}/100
                </p>
                <p className="text-gray-500 text-xs mt-1">
                  {result.detection.metrics.ai_score >= 70 ? 'High Risk' : 
                   result.detection.metrics.ai_score >= 50 ? 'Likely AI' :
                   result.detection.metrics.ai_score >= 30 ? 'Uncertain' : 'Low Risk'}
                </p>
              </div>
            </div>

            {/* AI Reasoning Card - Backup Reference */}
            {result.detection.ai_reasoning && (
              <div className="mt-8 p-6 rounded-lg bg-blue-500/10 border-2 border-blue-500">
                <div className="flex items-start gap-4">
                  <svg className="w-8 h-8 text-blue-500 shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-blue-400 mb-2">
                      🤖 AI Reasoning Verification (Backup)
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-gray-800/50 p-3 rounded">
                        <p className="text-xs text-gray-400">Verdict</p>
                        <p className={`text-lg font-bold ${
                          result.detection.ai_reasoning.verdict === 'REAL' ? 'text-green-400' :
                          result.detection.ai_reasoning.verdict === 'FAKE' ? 'text-red-400' :
                          'text-yellow-400'
                        }`}>
                          {result.detection.ai_reasoning.verdict}
                        </p>
                      </div>
                      <div className="bg-gray-800/50 p-3 rounded">
                        <p className="text-xs text-gray-400">Category</p>
                        <p className="text-lg font-bold text-blue-400">
                          {result.detection.ai_reasoning.sub_category.replace(/_/g, ' ')}
                        </p>
                      </div>
                      <div className="bg-gray-800/50 p-3 rounded">
                        <p className="text-xs text-gray-400">AI Confidence</p>
                        <p className="text-lg font-bold text-blue-400">
                          {result.detection.ai_reasoning.confidence_score}%
                        </p>
                      </div>
                    </div>
                    <div className="bg-gray-800/50 p-4 rounded mb-3">
                      <p className="text-sm font-semibold text-blue-300 mb-2">🔍 Primary Evidence:</p>
                      <p className="text-sm text-gray-300">{result.detection.ai_reasoning.primary_smoking_gun}</p>
                    </div>
                    <div className="bg-gray-800/50 p-4 rounded mb-3">
                      <p className="text-sm font-semibold text-blue-300 mb-2">💭 Reasoning Chain:</p>
                      <ol className="list-decimal list-inside space-y-1">
                        {result.detection.ai_reasoning.reasoning_chain.map((step, idx) => (
                          <li key={idx} className="text-xs text-gray-300">{step}</li>
                        ))}
                      </ol>
                    </div>
                    <div className="bg-gray-800/50 p-4 rounded">
                      <p className="text-sm font-semibold text-blue-300 mb-2">📝 Explanation:</p>
                      <p className="text-sm text-gray-300">{result.detection.ai_reasoning.human_readable_explanation}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-cyan-500/30 bg-gray-900/50 mt-20">
        <div className="container mx-auto px-4 py-6 text-center text-gray-400 text-sm">
          <p>
            AI-generated images exhibit Gaussian-like noise (β ≈ 2.0), while real photos show heavy-tailed distributions (β &lt; 1.5)
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

