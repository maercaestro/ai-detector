# AI Image Forensics - Detector

A modern web application for detecting AI-generated images using Azimuthal Power Spectrum analysis.

## 🚀 Project Structure

```
ai-detector/
├── backend/
│   ├── main.py              # FastAPI server with /analyze endpoint
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main React component
    │   ├── ChartComponent.jsx  # Chart visualization
    │   └── index.css        # Tailwind CSS v4
    ├── package.json
    └── vite.config.js
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

## 🔧 Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## 🚀 Running the Application

### Start the Backend

From the `backend` directory:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Start the Frontend

From the `frontend` directory:
```bash
npm run dev
```

The web app will be available at `http://localhost:5173`

## 🎯 Usage

1. Open the web app in your browser
2. Drag and drop an image or click to browse
3. Click "Analyze Image"
4. View the Azimuthal Power Spectrum chart
5. AI-generated images typically show anomalies in the red highlighted zone (frequency > 0.7)

## 🔬 How It Works

The application analyzes images using **Azimuthal Power Spectrum Analysis**:

1. **FFT Transformation**: Converts the image to frequency domain
2. **Radial Profile**: Computes average power at each frequency radius
3. **Detection**: AI-generated images often show artifacts in high-frequency regions (f > 0.7)

## 🎨 Features

- **Modern UI**: Dark cyberpunk-themed interface with Tailwind CSS v4
- **Drag & Drop**: Easy image upload with visual feedback
- **Interactive Chart**: Recharts visualization with highlighted detection zones
- **Real-time Analysis**: Fast FFT-based computation
- **CORS Enabled**: Seamless frontend-backend communication

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **SciPy**: FFT operations

### Frontend
- **React 19**: UI framework
- **Vite**: Build tool
- **Tailwind CSS v4**: Styling with new @import syntax
- **Recharts**: Data visualization

## 📊 API Endpoints

### POST /analyze
Analyzes an uploaded image for AI generation artifacts.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image)

**Response:**
```json
{
  "frequency": [0.0, 0.01, ..., 1.0],
  "power": [120.5, 118.3, ..., 95.2]
}
```

## 🔍 Understanding the Results

- **Frequency (X-axis)**: Normalized spatial frequency (0 to 1)
- **Power (Y-axis)**: Magnitude spectrum in dB
- **Detection Zone**: High-frequency region (f > 0.7) highlighted in red
- **AI Artifacts**: Unusual patterns or drops in the red zone may indicate AI generation

## 📝 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
