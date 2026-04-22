# UrbexBench

A benchmark for evaluating vision language models on urban exploration (urbex) image classification. This project tests how well different AI models can classify whether a location in satellite imagery is abandoned or not.

### Example Image
![](img/abandoned/50.339378_7.623978.png)

## Leaderboard

| Rank | Model | Parameters | Accuracy | Total |
|----- |-------------------------------|------|----|-----|
| 1🥇  | Gemini 3.1 Flash Lite Preview | -    | 92 | 100 |
| 2🥈  | Qwen3 VL                      | 30   | 89 | 100 |
| 3🥉  | Qwen3.5 Plus 02-15            | 397  | 87 | 100 |
| 4    | Qwen3.5 Flash 02-23           | 35   | 86 | 100 |
| 5    | Kimi K2.5                     | 1000 | 85 | 100 |
| 5    | Qwen3 VL                      | 235  | 85 | 100 |
| 5    | Gemma 3                       | 12   | 85 | 100 |
| 6    | MiMo V2 Omni                  | -    | 84 | 100 |
| 6    | Qwen3.5                       | 9    | 84 | 100 |
| 7    | Grok 4.1 Fast                 | -    | 82 | 100 |
| 8    | Gemma 3                       | 27   | 75 | 100 |
| 9    | Ministral 2512                | 14   | 65 | 100 |
| 10   | Ministral 2512                | 8    | 63 | 100 |
| 11   | Gemma 3                       | 4    | 56 | 100 |
| 12   | Ministral 2512                | 3    | 55 | 100 |

![](accuracy_vs_cost.svg)

## Setup

### Prerequisites
- Python 3.8+
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ruben-blok/urbexbench.git
cd urbexbench
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

### Running Evaluations

Evaluate all models that haven't been tested yet:
```bash
python scripts/evaluate.py
```

The script will:
- Load all models from `models.json`
- Test each model on all images in the `img/` directory
- Save predictions and results to `results.json`

### Viewing Statistics

Generate and display accuracy statistics:
```bash
python scripts/stats.py
```

## Models Configuration

Edit `models.json` to specify which models to evaluate. The file contains a list of model IDs available through OpenRouter.

## How It Works

1. **Image Encoding**: Images are converted to base64 and sent to the API
2. **Prompt**: Models receive the image and a simple question: "Is this location abandoned?"
3. **Parsing**: The model's response is parsed for '0' (not-abandoned) or '1' (abandoned)
4. **Results**: Predictions are stored with ground truth labels for accuracy calculation

## License

See [LICENSE](LICENSE) file for details.