# Speech Completion Prediction Meter

A real-time speech analysis tool that predicts speech completion likelihood using an innovative ecological species accumulation model approach. This application analyzes ongoing speech patterns through dynamic topic clustering and provides intelligent predictions about when a speaker might finish their statement.

## 🧠 Project Approach Overview

Our innovative approach treats speech completion estimation as a **topic evolution problem**, inspired by the species accumulation model from ecology. 

### Core Hypothesis
A speech nears completion when:
- The speaker exhausts topics in their knowledge scope or speech scope
- Topics become repetitive, indicating content saturation

### Ecological Analogy
- **Topics in speech** ≡ **Species in ecology**
- **Words/sentences** ≡ **Individual observations**

This biomimetic approach allows us to model speech completion using established ecological principles of resource depletion and population dynamics.

## 🔬 Technical Implementation

### Phase 0: Text Preprocessing

#### Coreference Resolution
- Resolve pronouns and references to maintain topic coherence
- Ensures semantic continuity across sentence boundaries

#### Sentence Grouping and Filtering
- **Conjunction Joining**: Merge sentences starting with conjunctions ("and", "but", "however") to their preceding sentences as continuations
- **Length Filtering**: Remove sentences shorter than 3 words to eliminate filler content and non-contributory short phrases

### Phase 1: Dynamic Topic Clustering

#### Semantic Embedding
- Convert sentences to high-dimensional vectors using **Sentence Transformers** (spaCy: all-mpnet-base-v2)
- Captures semantic meaning and contextual relationships

#### Linear Similarity Based Grouping
- **Cluster Assignment**: Compare new sentences against existing topic clusters
- **Threshold Matching**: Add to existing cluster if similarity > threshold
- **New Cluster Creation**: Create new cluster if no suitable match found

#### Adaptive Thresholding
- **Temporal Decay**: Similarity threshold decays over time to allow topic branching
- **Distinctiveness Preservation**: Maintains cluster separation while enabling evolution

#### Dynamic Representatives
- **Centroid Updates**: Continuously update cluster centroids as new sentences are added
- **Representative Evolution**: Clusters adapt to accommodate new semantic content

### Phase 2: Completion Estimation Algorithms

#### Staleness Metric
Quantifies topic abandonment and temporal decay patterns:

- **Base Score**: `inactive_topics_ratio = topics_inactive_15+ / total_topics`
- **Age Amplification**: Weighted by topic introduction time (normalized by factor of 100)
- **Trend Component**: 5-sentence moving window trend analysis
- **Adaptive Formula**: 
  ```
  staleness = base × (1.2 + age_factor) + 0.3 × trend
  ```
- **Bounded Output**: Capped at 1.0 to prevent overflow conditions

#### Saturation Metric  
Measures topic distribution imbalance and engagement decay:

- **Population Imbalance**: Standard deviation of topic sizes (higher deviation indicates saturation)
- **Vitality Decay**: Proportion of topics with no recent additions (<8 sentences)
- **Novelty Deficit**: Lack of new topics in recent 20-sentence window
- **Depth Stagnation**: Average topic complexity in last 5 sentences
- **Weighted Formula**: 
  ```
  saturation = 0.3×novelty + 0.3×depth + 0.2×imbalance + 0.2×vitality_decay
  ```

#### Speech Length Estimation
Dynamic length prediction based on topic introduction patterns:

- **Topic Rate**: `rate = topics_discovered / current_position`
- **Inverse Relationship**: `estimated_length = 3.8 / max(topic_rate, 0.01)`
- **Bounded Range**: 40-400 sentences to handle statistical edge cases

#### Final Progress Calculation
**Adaptive Weighting System**:
- **Early Speech** (low time_weight): Emphasizes saturation over staleness
- **Late Speech** (high time_weight): Emphasizes staleness over saturation
- **Exponential Decay Model**: 
  ```
  progress = 100 × (1 - e^(-2.5 × combined × position_ratio))
  ```
- **Safety Bounds**: Progress clamped between 1-99% to avoid false completion signals

## 📁 Project Structure

```
speech-completion-prediction/
├── backend/
│   ├── app-old.py
│   ├── app-new.py               # Run this one for backend
│   ├── req.txt                  # Python dependencies
│             
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── SpeechAnalyzer.jsx  # Main React component
│   │   ├── App.js                  # App entry point
│   │   └── index.js               # React DOM render
│   ├── package.json               # Node.js dependencies
│   ├── tailwind.config.js         # Tailwind CSS configuration
│   
├── data/
│   └──                           # Data storage directory
└── README.md                     # Project documentation
```

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://python.org/downloads/)
- **Node.js 16+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/)

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/harsh-vardhan3/speech-completion-prediction
cd speech-completion-prediction
```

### 2. Backend Setup

#### Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r req.txt
```


#### Test Backend
```bash
python app-new.py
```

Expected output:
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
Starting Flask backend...
Loading models...
spaCy model loaded: True
SentenceTransformer model loaded: True
FastCoref model loaded: True
Backend ready!
```

### 3. Frontend Setup

#### Install Additional Dependencies
```bash

npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```


### 4. Run the Application

#### Start Backend (Terminal 1)
```bash
cd backend
# Activate virtual environment if not already active
python app-new.py
```

#### Start Frontend (Terminal 2)
```bash
cd frontend
npm start
```

The application will open at `http://localhost:3000`

## 🔧 Configuration

### Backend Configuration
- **Port**: Backend runs on port 5000 by default
- **CORS**: Configured to accept requests from frontend
- **Model**: Uses sentence-transformers for NLP processing

### Frontend Configuration
- **API Endpoint**: Configure in your component to point to `http://localhost:5000`
- **Styling**: Tailwind CSS for responsive design
- **Components**: Modular React components for easy maintenance

## 📊 Usage

1. **Start the Application**: Follow the setup steps above
2. **Access Interface**: Open `http://localhost:3000` in your browser
3. **Begin Analysis**: Use the speech input interface to start analysis
4. **View Predictions**: Monitor real-time completion predictions
5. **Review Results**: Analyze completion patterns and accuracy


## 🚨 Troubleshooting

### Common Issues

**Backend won't start:**
- Ensure Python virtual environment is activated
- Check all dependencies are installed: `pip install -r req.txt`
- Verify Python version is 3.8+

**Frontend build errors:**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version is 16+

**CORS Issues:**
- Ensure Flask-CORS is installed and configured
- Check that backend is running on the expected port

**Model Loading Issues:**
- First run may take time to download transformer models
- Ensure stable internet connection for initial model download

