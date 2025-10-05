# 🚀 HabitatLab - Space Habitat Communication Simulator

**NASA Space Apps Challenge 2025**  
*Optimizing communication networks for future lunar and Martian habitats*

---

## 📖 Overview

HabitatLab is a comprehensive simulation system designed to model and optimize wireless communication networks in space habitats. The system analyzes network topology, signal quality, and resilience to ensure reliable communications between critical modules like life support, power, medical, and communication systems.

![HabitatLab Simulation](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps-orange)

---

## 🎯 Key Features

### 📡 Communication Modeling
- **AWGN Channel Simulation** - Realistic signal transmission with noise
- **5G/6G Technology Comparison** - Analyze different communication technologies
- **Path Loss Calculations** - Distance-based signal degradation modeling
- **BER/PER Analysis** - Bit Error Rate and Packet Error Rate calculations

### 🏗️ Network Analysis
- **Topology Management** - Create and manage habitat module networks
- **Quality Matrix** - Packet Error Rate for all module pairs
- **Failure Simulation** - Test network resilience to equipment failures
- **Connectivity Assessment** - Ensure critical communications remain active

### 🖥️ User Interface
- **Interactive Canvas** - Drag-and-drop module placement
- **Real-time Metrics** - Area, module count, connection status
- **Visual Analytics** - Heatmaps and quality visualization
- **Export Capabilities** - JSON configurations and CSV reports

### 🔬 Advanced Features
- **ML Integration** - Object detection and classification with OpenCV
- **Network Resilience** - Alternative path finding and redundancy analysis
- **Performance Benchmarking** - Compare different habitat configurations

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/habitatlab.git
cd habitatlab

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
