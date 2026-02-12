I am building a new project that simulates a real-world industrial computer vision system similar to pizza quality control in factories.

The goal is to design and implement a production-style ML system that:

• trains a computer vision model to classify or detect product quality (good vs defective)
• fine-tunes pretrained deep learning models (PyTorch or TensorFlow)
• serves inference through FastAPI
• processes images in real-time through a Kafka streaming pipeline
• is optimized to run on CPU or low-resource edge devices
• uses only open-source tools (no paid APIs or services)

This project is NOT a research notebook. It must look like a production ML system with clean architecture and services.

====================================================

PROJECT GOALS

====================================================

I want to simulate this pipeline:

Camera → Kafka Producer → Kafka Topic → ML Consumer → Model Inference → API → Database/Output

The system should:

1) Train a vision model
   - use transfer learning (ResNet/EfficientNet/MobileNet)
   - fine-tune last layers
   - save weights
   - export model for inference

2) Serve the model with FastAPI
   - POST /predict (image upload)
   - returns label + confidence
   - optimized for CPU inference

3) Add Kafka streaming
   - producer sends image paths or image bytes
   - consumer loads model and runs inference
   - results saved to DB or logs

4) Be modular and production-like
   - separate training, inference, service, streaming
   - clean folder structure
   - device-agnostic code (CPU/GPU)
   - Docker support

5) Demonstrate ML engineering best practices
   - data cleaning
   - augmentation
   - batching
   - logging
   - model serialization
   - reproducibility

====================================================

WHAT I NEED YOU TO GENERATE

====================================================

Please generate:

1) High-level system architecture explanation
2) Recommended folder structure
3) Python code for:
   • model.py (model definition)
   • train.py (training + fine-tuning)
   • inference.py (prediction wrapper)
   • FastAPI service (api.py)
   • Kafka producer
   • Kafka consumer
   • config files
4) requirements.txt
5) example dataset suggestions
6) instructions to run locally step-by-step
7) comments explaining design choices

====================================================

TECH STACK CONSTRAINTS

====================================================

Use only:

Python
PyTorch or TensorFlow
FastAPI
Kafka (kafka-python or confluent-kafka)
OpenCV or Pillow
Pandas/Numpy
Docker (optional)

NO paid APIs.
NO cloud dependencies.

Everything must run locally.

====================================================

STYLE REQUIREMENTS

====================================================

Output clean, production-quality code.

Each module should:
- be independent
- well commented
- typed when possible
- easy to test

Explain WHY each component exists and how it mimics real-world ML engineering.

Focus on practical deployment, not theory.

====================================================

GOAL

====================================================

This project should showcase skills required for ML Engineer roles:
• deep learning
• model fine-tuning
• deployment
• streaming systems
• FastAPI services
• real-time inference
• edge optimization

Treat this as if you are mentoring me step-by-step to build it from scratch.
Start by generating the project structure and training pipeline first.



1. Build model (the brain)
PyTorch
fine-tuning
datasets
training

Output: saving weights


2. Inference wrapper (make model usable)
loading weights
preprocessing
prediction function
CPU optimization

Output: predictor.predict(image)



quality-inspector/
│
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env
│
├── configs/
│   ├── model.yaml
│   ├── kafka.yaml
│   └── app.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   └── exploration.ipynb
│
├── models/
│   └── weights.pt
│
├── src/
│   │
│   ├── common/                 # shared utilities
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── utils.py
│   │
│   ├── training/               # ONLY training code
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── inference/              # reusable inference logic
│   │   ├── predictor.py
│   │   ├── preprocess.py
│   │   └── postprocess.py
│   │
│   ├── service/                # FastAPI service
│   │   ├── api.py
│   │   ├── schemas.py
│   │   └── dependencies.py
│   │
│   ├── streaming/              # Kafka pipeline
│   │   ├── producer.py
│   │   ├── consumer.py
│   │   └── topics.py
│   │
│   └── db/
│       ├── models.py
│       └── repository.py
│
├── scripts/
│   ├── train.sh
│   ├── run_api.sh
│   └── run_stream.sh
│
└── tests/
    ├── test_inference.py
    └── test_api.py
