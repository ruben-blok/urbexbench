# UrbexBench

A benchmark for evaluating vision language models on urban exploration (urbex) image classification. This project tests how well different AI models can classify whether a location in satellite imagery is abandoned or not.

### Example Image
![](img/abandoned/50.339378_7.623978.png)

## Leaderboard

| Rank | Model | Parameters | Accuracy |
|----- |-------------------------------|------|-------|
| 1🥇  | Gemini 3.7 Flash              | -    | 74,5% |
| 2🥈  | Gemini 3.1 Flash Lite Preview | -    | 71%   |
| 3🥉  | Gemini 3.1 Flash Lite         | -    | 70%   |
| 4    | Qwen3 VL                      | 30   | 69%   |
| 5    | DeepSeek V4 Flash Vision      | -    | 68,5% |
| 6    | Gemini 2.5 Flash Lite         | -    | 68%   |
| 6    | Qwen3.5 35B-A3B               | 35   | 68%   |
| 8    | Ox Alpha                      | -    | 66,5% |
| 8    | Qwen3 VL                      | 32   | 66,5% |
| 8    | Qwen3.5                       | 27   | 66,5% |
| 11   | Nemotron 3 Nano Omni          | 30   | 66%   |
| 12   | Nemotron Nano 12B V2 VL       | 12   | 65,5% |
| 12   | MiMo V2.5                     | -    | 65,5% |
| 14   | Gemma 3                       | 27   | 64,5% |
| 14   | Gemma 3                       | 12   | 64,5% |
| 14   | Qwen3.5                       | 9    | 64,5% |
| 17   | Qwen3 VL                      | 8    | 64%   |
| 18   | Gemma 3                       | 4    | 62,5% |
| 18   | Gemma 4                       | 26   | 62,5% |
| 20   | Gemma 4                       | 31   | 59,5% |

\* Evaluations run with reasoning disabled where possible. Gemini 3.7 Flash and Ox Alpha force reasoning on their endpoints and were evaluated at the lowest supported effort (`low` / `minimal`).

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