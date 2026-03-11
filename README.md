# UrbexBench

A benchmark for evaluating vision language models on urban exploration (urbex) image classification. This project tests how well different AI models can classify whether a location in satellite imagery is abandoned or not.

## Leaderboard

| Rank | Model | Accuracy | Correct | Total |
|------|-------|----------|---------|-------|
| 🥇 1 | google/gemini-3.1-flash-lite-preview | 92.00% | 92 | 100 |
| 🥈 2 | qwen/qwen3-vl-30b-a3b-thinking | 89.00% | 89 | 100 |
| 🥉 3 | qwen/qwen3.5-plus-02-15 | 87.00% | 87 | 100 |
| 4 | qwen/qwen3.5-flash-02-23 | 86.00% | 86 | 100 |
| 5 | qwen/qwen3-vl-235b-a22b-thinking | 85.00% | 85 | 100 |
| 5 | moonshotai/kimi-k2.5 | 85.00% | 85 | 100 |
| 5 | google/gemma-3-12b-it | 85.00% | 85 | 100 |
| 8 | x-ai/grok-4.1-fast | 82.00% | 82 | 100 |
| 9 | google/gemma-3-27b-it | 75.00% | 75 | 100 |
| 10 | google/gemma-3-4b-it | 56.00% | 56 | 100 |

## Project Structure

```
urbexbench/
├── evaluate.py           # Main script to evaluate models
├── stats.py             # Generate statistics from results
├── models.json          # Configuration of models to test
├── results.json         # Stores evaluation results
├── img/
│   ├── abandoned/       # Test images of abandoned locations
│   └── not-abandoned/   # Test images of occupied locations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

### Prerequisites
- Python 3.8+
- OpenRouter API key ([get one here](https://openrouter.ai))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ruben-blok/urbexbench.git
cd urbexbench
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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
python evaluate.py
```

The script will:
- Load all models from `models.json`
- Test each model on all images in the `img/` directory
- Save predictions and results to `results.json`

### Viewing Statistics

Generate and display accuracy statistics:
```bash
python stats.py
```

Example output:
```
======================================================================
MODEL ACCURACY STATISTICS
======================================================================
Model                                         Correct    Total      Accuracy  
----------------------------------------------------------------------
google/gemini-3.1-flash-lite-preview          92         100        92.00%
qwen/qwen3-vl-30b-a3b-thinking                89         100        89.00%
qwen/qwen3.5-plus-02-15                       87         100        87.00%
qwen/qwen3.5-flash-02-23                      86         100        86.00%
qwen/qwen3-vl-235b-a22b-thinking              85         100        85.00%
moonshotai/kimi-k2.5                          85         100        85.00%
google/gemma-3-12b-it                         85         100        85.00%
x-ai/grok-4.1-fast                            82         100        82.00%
google/gemma-3-27b-it                         75         100        75.00%
google/gemma-3-4b-it                          56         100        56.00%
----------------------------------------------------------------------
Total models evaluated: 10
```

## Models Configuration

Edit `models.json` to specify which models to evaluate. The file contains a list of model IDs available through OpenRouter.

## Test Images

Place test images in:
- `img/abandoned/` - Images of abandoned locations
- `img/not-abandoned/` - Images of occupied locations

Images should be in PNG format. The classifier uses a simple prompt asking models to determine if a location is abandoned or not.

## How It Works

1. **Image Encoding**: Images are converted to base64 and sent to the API
2. **Prompt**: Models receive the image and a simple question: "Is this location abandoned?"
3. **Parsing**: The model's response is parsed for '0' (not-abandoned) or '1' (abandoned)
4. **Results**: Predictions are stored with ground truth labels for accuracy calculation

## Results Format

Results are saved in `results.json` with the following structure:
```json
{
  "model_results": [
    {
      "model": "google/gemini-3.1-flash-lite-preview",
      "predictions": {
        "abandoned": [
          {
            "image": "image_name.png",
            "prediction": "abandoned",
            "answer": "abandoned"
          }
        ],
        "not-abandoned": [...]
      }
    }
  ]
}
```

## Requirements

See `requirements.txt` for all dependencies. Key requirements:
- `openai` - OpenAI SDK for API access
- `python-dotenv` - Environment variable management
- `tqdm` - Progress bar for evaluations

## Notes

- The evaluation script skips models that have already been tested to avoid duplicate API calls
- Rate limiting may occur with OpenRouter; the script handles 429 errors gracefully
- Ground truth labels are determined by the subdirectory structure (`abandoned/` vs `not-abandoned/`)

## License

See [LICENSE](LICENSE) file for details.