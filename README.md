# Pure Python Spam Email Detector (Naive Bayes)

A spam classifier built from scratch using Naive Bayes. The core math (tokenizing, probability, smoothing) is plain Python — only `math`, `re`, `collections`. `pandas` is just used to load/split the dataset, and `colorama` for colored CLI output.

## Features

- **Leetspeak fix** — turns `Fr33`, `Cl1ck`, `b0nus` back into normal words before processing.
- **Bigrams** — looks at word pairs (`click_here`) not just single words, so it catches phrase context.
- **Log probabilities** — adds log-probs instead of multiplying raw probs (avoids underflow on long messages).
- **Laplace smoothing** — no zero-probability errors on words it hasn't seen before.
- **Auto train/test split** — trains on 80% of ~5,500 messages, tests on the rest, prints accuracy.

## Structure

```
├── src/
│   └── detector.py       # AdvancedSpamDetector (bigrams, leetspeak, math)
├── main.py               # loads data, runs test eval, interactive CLI
├── requirements.txt      # pandas, colorama
└── README.md
```

## Quick Start

```bash
git clone https://github.com/eyuel-stack/pure-python-spam-detector.git
cd pure-python-spam-detector
pip install -r requirements.txt
python main.py
```

Trains the model, prints accuracy, then lets you type messages to classify live.

## Using It in Code

```python
from src.detector import AdvancedSpamDetector

detector = AdvancedSpamDetector()
detector.train(messages, labels) 
detector.predict("Cl1ck here for your FR33 b0nus now!!!")

detector.predict_log_proba("Cl1ck here for your FR33 b0nus now!!!")
```

## Why It Beats Basic Bag-of-Words

| Input | Basic Bag-of-Words NB | This Version |
|---|---|---|
| `Cl1ck h3re for your fr33 b0nus` | words look unseen/rare, weak signal | normalized to real words first, strong spam signal |
| "click" in a phishing link vs a photo tutorial | same weight either way, confusing | bigrams (`click_here` vs `click_shutter`) tell them apart |
| Long messages | probabilities can underflow to 0 | log-sum keeps it stable |
| Word not seen in training | prediction can break (zero prob) | smoothing keeps it working |

## Requirements

```
pandas
colorama
```

Rest is pure Python.