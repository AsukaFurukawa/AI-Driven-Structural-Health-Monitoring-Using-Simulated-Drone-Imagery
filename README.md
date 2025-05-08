# AI-Driven Structural Health Monitoring Using Simulated Drone Imagery

This project implements an AI-powered structural health monitoring system that uses simulated drone imagery to detect and analyze cracks in structures. The system combines computer vision and deep learning techniques to provide real-time monitoring and analysis capabilities.

## Features

- Real-time crack detection using deep learning models
- Web-based interface for image upload and analysis
- Real-time training progress monitoring via WebSocket
- Support for both simulated and real drone imagery
- Comprehensive model evaluation and visualization tools
- RESTful API for integration with other systems

## Project Structure

```
AI-Driven-Structural-Health-Monitoring-Using-Simulated-Drone-Imagery/
├── config/
│   └── config.yaml           # Configuration settings
├── data/
│   ├── raw/                 # Raw dataset
│   └── processed/           # Processed dataset
├── notebooks/
│   └── analysis.ipynb       # Data analysis and visualization
├── src/
│   ├── api/                 # API server
│   ├── data/               # Data processing
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── checkpoints/            # Model checkpoints
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Driven-Structural-Health-Monitoring-Using-Simulated-Drone-Imagery.git
cd AI-Driven-Structural-Health-Monitoring-Using-Simulated-Drone-Imagery
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
cd src/api
uvicorn app:app --reload
```

The API server will be available at `http://localhost:8000`.

### API Endpoints

- `POST /predict`: Upload and analyze an image
- `POST /train`: Start model training
- `GET /metrics`: Get model performance metrics
- `GET /`: Web interface for image upload and analysis
- `WS /ws/training`: WebSocket endpoint for real-time training updates

### Web Interface

Access the web interface at `http://localhost:8000` to:
- Upload images for analysis
- View detection results
- Monitor training progress
- Access model metrics and visualizations

## Model Training

1. Prepare your dataset in the `data/raw` directory
2. Run the data processing script:
```bash
python src/data/process_data.py
```

3. Start training:
```bash
python src/models/train.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses Black for code formatting and Flake8 for linting:

```bash
black .
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for the amazing tools and libraries

## Contact

For questions and support, please open an issue in the GitHub repository.