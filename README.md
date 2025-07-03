# Sinhala LLM Project

A Sinhala language model with web interface for text generation, chat, and completion.

## Project Structure

```
sinhalallm/
├── src/                         # Source code
│   ├── model_architecture.py    # Model architecture definition
│   ├── use_model.py             # Model inference engine
│   ├── train_model.py           # Model training script
│   ├── data_preparation.py      # Data preprocessing
│   ├── tokenizer_training.py    # Tokenizer training
│   └── web/                     # Web interface
│       ├── web_interface.py     # Flask web application
│       ├── start_web.py         # Web server starter
│       └── templates/           # HTML templates
│           └── index.html
├── models/                      # Trained models and tokenizer
│   ├── sinhala_llm_best.pt      # Best trained model
│   ├── sinhala_llm_latest.pt    # Latest model checkpoint
│   └── tokenizer/               # Tokenizer files
│       ├── sinhala_sp.vocab
│       ├── sinhala_sp.model
│       └── sinhala_tokenizer/
├── data/                        # Training data
│   ├── raw/                     # Raw dataset files
│   │   ├── Sinhala-QA-10K.csv
│   │   ├── Sinhala-QA-1K.csv
│   │   ├── Sinhala text-Summary.csv
│   │   └── databricks-dolly-15k-sinhala.csv
│   └── processed/               # Processed training data
│       ├── sinhala_corpus.txt
│       ├── train_alpaca.json
│       ├── val_alpaca.json
│       ├── train_conversations.txt
│       └── val_conversations.txt
├── scripts/                     # Utility scripts
│   └── quick_demo.py           # Quick demonstration script
├── logs/                        # Log files
│   └── training.log
└── requirements.txt             # Python dependencies
```

## Usage

### Quick Demo
```bash
cd scripts
python quick_demo.py
```

### Web Interface
```bash
cd src/web
python start_web.py
```

### Command Line Interface
```bash
cd src
python use_model.py --chat
```

### Training
```bash
cd src
python train_model.py
```

## Features

- 📝 **Text Generation**: Generate creative Sinhala text from prompts
- 💬 **Chat Mode**: Interactive conversation in Sinhala
- ✏️ **Text Completion**: Complete partial Sinhala text
- 🌐 **Web Interface**: User-friendly web application
- 🔧 **Training Pipeline**: Complete model training workflow 