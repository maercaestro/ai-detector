import React from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

const ChartComponent = ({ data }) => {
  // Check if we have noise distribution data
  if (!data.detection.noise_distribution) {
    return <div className="text-gray-400">No distribution data available</div>;
  }

  const { histogram, ggd_fit, gaussian_ref } = data.detection.noise_distribution;
  
  // Combine histogram and curves into chart data
  const chartData = histogram.bins.map((bin, idx) => ({
    noise: bin,
    histogram: histogram.values[idx],
    ggd_fit: ggd_fit.y[Math.floor(idx * (ggd_fit.y.length / histogram.bins.length))],
    gaussian: gaussian_ref.y[Math.floor(idx * (gaussian_ref.y.length / histogram.bins.length))],
  }));

  const ggdShape = data.detection.metrics.ggd_shape;
  const isAI = data.detection.is_ai_generated;

  return (
    <div className="w-full h-[500px] bg-gray-900 rounded-lg border border-cyan-500/30 p-6">
      <h2 className="text-2xl font-bold text-cyan-400 mb-4">
        Noise Distribution Analysis (GGD Shape β = {ggdShape.toFixed(3)})
      </h2>
      <ResponsiveContainer width="100%" height="90%">
        <ComposedChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          
          <XAxis
            dataKey="noise"
            stroke="#06b6d4"
            label={{
              value: 'Noise Residual Value',
              position: 'insideBottom',
              offset: -5,
              fill: '#06b6d4',
            }}
            tick={{ fill: '#9ca3af' }}
            domain={[-50, 50]}
          />
          
          <YAxis
            stroke="#06b6d4"
            label={{
              value: 'Probability Density',
              angle: -90,
              position: 'insideLeft',
              fill: '#06b6d4',
            }}
            tick={{ fill: '#9ca3af' }}
          />
          
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #06b6d4',
              borderRadius: '8px',
              color: '#fff',
            }}
            labelStyle={{ color: '#06b6d4' }}
          />
          
          <Legend 
            wrapperStyle={{ 
              color: '#9ca3af',
              paddingTop: '10px'
            }}
            iconType="line"
            verticalAlign="bottom"
            height={36}
          />
          
          {/* Reference line at x = 0 (mean) */}
          <ReferenceLine
            x={0}
            stroke="#6b7280"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          
          {/* Histogram bars */}
          <Bar
            dataKey="histogram"
            fill="#06b6d4"
            fillOpacity={0.6}
            name="Noise Histogram"
          />
          
          {/* GGD Fit curve */}
          <Line
            type="monotone"
            dataKey="ggd_fit"
            stroke={isAI ? '#ef4444' : '#10b981'}
            strokeWidth={3}
            dot={false}
            name={`GGD Fit (β=${ggdShape.toFixed(2)})`}
          />
          
          {/* Gaussian reference curve */}
          <Line
            type="monotone"
            dataKey="gaussian"
            stroke="#fbbf24"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Gaussian Ref (β=2.0)"
          />
        </ComposedChart>
      </ResponsiveContainer>
      
      <div className="mt-4 flex gap-6 text-sm flex-wrap">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-cyan-500/60 border border-cyan-500 rounded"></div>
          <span className="text-gray-300">Noise Histogram</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded ${isAI ? 'bg-red-500' : 'bg-green-500'}`}></div>
          <span className="text-gray-300">GGD Fit (β={ggdShape.toFixed(2)})</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-yellow-500"></div>
          <span className="text-gray-300">Gaussian Reference (AI-like)</span>
        </div>
        <div className="ml-auto text-gray-400 italic">
          {ggdShape < 1.5 ? 'Heavy-tailed (Real Photo)' : ggdShape < 1.8 ? 'Moderate' : 'Gaussian-like (AI)'}
        </div>
      </div>
    </div>
  );
};

export default ChartComponent;
