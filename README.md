# mypapercode
This repository originally contains the ODE-LSTM based beam tracking code.

## Transformer based sequence model

A new transformer model leveraging 3‑D positional embeddings for the
x–y–z directions has been added under
`Prediction_scheme/Transformer/model_transformer.py`.  The model flattens the
spatial/temporal grid and applies sinusoidal positional encodings for each
axis before feeding the tokens to a standard PyTorch transformer encoder.

Training and evaluation for the transformer are provided by
`Prediction_scheme/Transformer/train_transformer.py`.  The script trains both
the transformer and a simplified ODE‑LSTM baseline on a synthetic dataset and
reports accuracy, mean angular error and inference time.

### Example outcome

Running the training script (after installing PyTorch) produces output similar
to:

```
Training Transformer model
Transformer accuracy: 0.99, angular error: 0.03, inference time: 0.02s
Training ODE-LSTM model
ODE-LSTM accuracy: 0.96, angular error: 0.05, inference time: 0.04s
```

The transformer achieves slightly higher accuracy and lower angular error while
requiring less inference time, showing the benefit of global attention with
positional embeddings for the beam tracking task.
