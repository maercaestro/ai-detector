import React from 'react';

const HeatmapComponent = ({ data }) => {
  // Debug logging
  console.log('HeatmapComponent received data:', data);
  console.log('detection:', data?.detection);
  console.log('local_consistency:', data?.detection?.local_consistency);
  
  // Access local_consistency from data.detection
  const localConsistency = data?.detection?.local_consistency;
  
  if (!data || !localConsistency || !localConsistency.heatmap) {
    return (
      <div className="bg-gray-900 rounded-lg border border-cyan-500/30 p-6">
        <h3 className="text-xl font-bold text-cyan-400 mb-4">
          Local Consistency Heatmap
        </h3>
        <div className="text-gray-500 text-center py-8">
          {!data ? 'No data available' :
           !localConsistency ? 'No local consistency data in detection' :
           !localConsistency.heatmap ? 'No heatmap data' :
           'Loading...'}
        </div>
      </div>
    );
  }

  const { heatmap, suspicious_patches, blocks_analyzed, max_beta, min_beta } = localConsistency;
  
  // Get dimensions
  const rows = heatmap.length;
  const cols = rows > 0 ? heatmap[0].length : 0;
  
  if (rows === 0 || cols === 0) {
    return null;
  }

  // Calculate cell size based on container
  const maxSize = 400;
  const cellSize = Math.floor(maxSize / Math.max(rows, cols));

  // Color mapping function (β value to color)
  const getColor = (beta) => {
    if (beta === 0) return '#1f2937'; // Gray for invalid
    
    // Real (green) β < 1.0
    // Uncertain (yellow) 1.0 <= β < 1.5
    // AI (red) β >= 1.5
    
    if (beta < 1.0) {
      // Green gradient (darker to brighter)
      const intensity = Math.min(beta / 1.0, 1);
      return `rgb(${Math.floor(16 + intensity * 50)}, ${Math.floor(128 + intensity * 50)}, ${Math.floor(56 + intensity * 50)})`;
    } else if (beta < 1.5) {
      // Yellow gradient
      const intensity = (beta - 1.0) / 0.5;
      return `rgb(${Math.floor(200 + intensity * 39)}, ${Math.floor(180 - intensity * 20)}, ${Math.floor(40 - intensity * 40)})`;
    } else {
      // Red gradient (gets brighter with higher β)
      const intensity = Math.min((beta - 1.5) / 0.7, 1);
      return `rgb(${Math.floor(180 + intensity * 59)}, ${Math.floor(48 - intensity * 28)}, ${Math.floor(48 - intensity * 28)})`;
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-cyan-500/30 p-6">
      <h3 className="text-xl font-bold text-cyan-400 mb-4">
        Local Consistency Heatmap
      </h3>
      
      <div className="flex flex-col items-center gap-4">
        {/* Heatmap Grid */}
        <div 
          className="border border-gray-700 rounded overflow-hidden"
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${rows}, ${cellSize}px)`,
            gap: '1px',
            backgroundColor: '#374151'
          }}
        >
          {heatmap.map((row, i) =>
            row.map((beta, j) => (
              <div
                key={`${i}-${j}`}
                style={{
                  backgroundColor: getColor(beta),
                  width: `${cellSize}px`,
                  height: `${cellSize}px`,
                }}
                title={`Block [${i},${j}]: β=${beta.toFixed(3)}`}
                className="hover:opacity-80 transition-opacity"
              />
            ))
          )}
        </div>

        {/* Legend */}
        <div className="w-full max-w-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Real (Natural)</span>
            <span className="text-xs text-gray-400">Uncertain</span>
            <span className="text-xs text-gray-400">AI (Synthetic)</span>
          </div>
          <div className="h-4 rounded-full overflow-hidden flex">
            <div className="flex-1" style={{ background: 'linear-gradient(to right, rgb(16, 128, 56), rgb(66, 178, 106))' }}></div>
            <div className="flex-1" style={{ background: 'linear-gradient(to right, rgb(200, 180, 40), rgb(239, 160, 0))' }}></div>
            <div className="flex-1" style={{ background: 'linear-gradient(to right, rgb(239, 68, 68), rgb(220, 38, 38))' }}></div>
          </div>
          <div className="flex items-center justify-between mt-1">
            <span className="text-xs text-gray-500">β &lt; 1.0</span>
            <span className="text-xs text-gray-500">β ≈ 1.0-1.5</span>
            <span className="text-xs text-gray-500">β &gt; 1.5</span>
          </div>
        </div>

        {/* Stats */}
        <div className="w-full grid grid-cols-2 gap-4 text-sm">
          <div className="bg-gray-950/50 rounded p-3">
            <div className="text-gray-400 text-xs">Blocks Analyzed</div>
            <div className="text-white text-lg font-bold">{blocks_analyzed}</div>
          </div>
          <div className="bg-gray-950/50 rounded p-3">
            <div className="text-gray-400 text-xs">Suspicious Patches</div>
            <div className="text-red-400 text-lg font-bold">
              {suspicious_patches} ({((suspicious_patches/blocks_analyzed)*100).toFixed(0)}%)
            </div>
          </div>
          <div className="bg-gray-950/50 rounded p-3">
            <div className="text-gray-400 text-xs">Min β</div>
            <div className="text-green-400 text-lg font-bold">{min_beta.toFixed(3)}</div>
          </div>
          <div className="bg-gray-950/50 rounded p-3">
            <div className="text-gray-400 text-xs">Max β</div>
            <div className="text-red-400 text-lg font-bold">{max_beta.toFixed(3)}</div>
          </div>
        </div>

        <div className="text-xs text-gray-500 text-center">
          Each cell represents a 64×64 pixel patch analyzed for noise consistency
        </div>
      </div>
    </div>
  );
};

export default HeatmapComponent;
