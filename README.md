# FlashDeBERTa 🦾 – Boost inference speed by 3-5x ⚡ and run DeBERTa models on long sequences 📚.

**FlashDeBERTa** is an optimized version of the DeBERTa model leveraging flash attention to implement a disentangled attention mechanism. It significantly reduces memory usage and latency, especially with long sequences. The project enables loading and running original DeBERTa models on tens of thousands of tokens without retraining, maintaining original accuracy.

### Use Cases

DeBERTa remains one of the top-performing models for the following tasks:

- **Named Entity Recognition:** It serves as the main backbone for models such as [GLiNER](https://github.com/urchade/GLiNER), an efficient architecture for zero-shot information extraction.
- **Text Classification:** DeBERTa is highly effective for supervised and zero-shot classification tasks, such as [GLiClass](https://github.com/Knowledgator/GLiClass).
- **Reranking:** The model offers competitive performance compared to other reranking models, making it a valuable component in many RAG systems.

> [!warning]
> This project is under active development and may contain bugs. Please create an issue if you encounter bugs or have suggestions for improvements.

### Installation

First, install the package:

```bash
pip install flashdeberta -U
```

Then import the appropriate model heads for your use case and initialize the model from pretrained checkpoints:

```python
from flashdeberta import FlashDebertaV2Model  # FlashDebertaV2ForSequenceClassification, FlashDebertaV2ForTokenClassification, etc.
from transformers import AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = FlashDebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to('cuda')

# Tokenize input text
input_text = "Hello world!"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')

# Model inference
outputs = model(input_ids)
```

In order to switch to eager attention implementation, initialise a model in the following way:
```python
model = FlashDebertaV2Model.from_pretrained("microsoft/deberta-v3-base", _attn_implementation='eager').to('cuda')
```

### Benchmarks

While context-to-position and position-to-context biases still require quadratic memory, our flash attention implementation reduces overall memory requirements to nearly linear. This efficiency is particularly impactful for longer sequences. Starting from 512 tokens, FlashDeBERTa achieves more than a 50% performance improvement, and at 4k tokens, it's over 5 times faster than naive implementations.

![benchmarking](images/benchmarking.png)

### Future Work

- Implement backward kernels.
- Train DeBERTa models on 8,192-token sequences using high-quality data.
- Integrate FlashDeBERTa into GLiNER and GLiClass.
- Train multi-modal DeBERTa models.

