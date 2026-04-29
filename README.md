# 🔄 Transformer Model from Scratch using TensorFlow

> A complete implementation of the Transformer architecture from first principles using TensorFlow/Keras

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

---

## 📑 Quick Navigation

<details open>
<summary><b>🎯 Table of Contents</b></summary>

- [Overview](#-overview)
- [Architecture Flow](#-architecture-flow)
- [Core Components](#-core-components)
- [Detailed Architecture](#-detailed-architecture)
- [Implementation Guide](#-implementation-guide)
- [Usage Example](#-usage-example)
- [Key Concepts](#-key-concepts)
- [Project Structure](#-project-structure)

</details>

---

## 🎯 Overview

The **Transformer** is a deep learning architecture designed for sequence-to-sequence tasks such as:
- Machine Translation
- Text Summarization
- Question Answering
- Language Modeling

Unlike RNNs/LSTMs, Transformers use **self-attention mechanisms** to capture long-range dependencies efficiently and enable parallel processing.

### Key Advantages
✅ Parallel processing of sequences  
✅ Captures long-range dependencies  
✅ Highly parallelizable training  
✅ State-of-the-art results on NLP tasks  

---

## 🔀 Architecture Flow

```mermaid
graph TD
    Input["📥 Input Sequence"] --> Embed["Embedding Layer"]
    Embed --> PosEnc["Positional Encoding"]
    PosEnc --> EncStack["Encoder Stack<br/>(N layers)"]
    
    EncStack --> EncOut["Encoder Output"]
    
    EncOut --> DecStack["Decoder Stack<br/>(N layers)"]
    Target["📥 Target Sequence"] --> TgtEmbed["Embedding Layer"]
    TgtEmbed --> TgtPos["Positional Encoding"]
    TgtPos --> DecStack
    
    DecStack --> DecOut["Decoder Output"]
    DecOut --> Linear["Linear Layer"]
    Linear --> Output["📤 Output Logits"]
    Output --> Softmax["Softmax"]
    Softmax --> Result["🎯 Predictions"]
    
    style Input fill:#e1f5ff
    style Result fill:#c8e6c9
    style EncOut fill:#fff9c4
    style DecOut fill:#fff9c4
```

---

## 🧩 Core Components

<details open>
<summary><b>Core Building Blocks</b></summary>

### 1. **Positional Encoding** 
Encodes token positions using sine/cosine functions.
```
Position (pos, 2i):     sin(pos / 10000^(2i/d_model))
Position (pos, 2i+1):   cos(pos / 10000^(2i/d_model))
```

### 2. **Multi-Head Attention**
Applies attention mechanism with multiple representation subspaces.

### 3. **Feed-Forward Network**
Position-wise dense layers: Dense → ReLU → Dense

### 4. **Layer Normalization & Residual Connections**
Stabilizes training with: `Output = LayerNorm(Input + Sublayer(Input))`

</details>

---

## 📊 Detailed Architecture

### Complete Transformer Block Diagram

```mermaid
graph TB
    subgraph Encoder["🔵 ENCODER"]
        EmbE["Embedding"]
        PosE["Positional Encoding"]
        EncBlock["Transformer Block × N"]
        
        EmbE --> PosE
        PosE --> EncBlock
        
        subgraph EncDetails["Transformer Block"]
            MHA1["Multi-Head<br/>Attention"]
            LN1["Layer Norm"]
            Add1["+"]
            FFN1["Feed-Forward<br/>Network"]
            LN2["Layer Norm"]
            Add2["+"]
            Dropout1["Dropout"]
            Dropout2["Dropout"]
            
            MHA1 --> Dropout1
            Dropout1 --> Add1
            LN1 --> Add1
            Add1 --> FFN1
            FFN1 --> Dropout2
            Dropout2 --> Add2
            LN2 --> Add2
        end
        
        EncBlock --> EncOut["Encoder Output"]
    end
    
    subgraph Decoder["🟠 DECODER"]
        EmbD["Embedding"]
        PosD["Positional Encoding"]
        DecBlock["Transformer Block × N"]
        
        EmbD --> PosD
        PosD --> DecBlock
        
        subgraph DecDetails["Decoder Block"]
            MHA2["Masked Multi-Head<br/>Attention"]
            LN3["Layer Norm"]
            Add3["+"]
            CMHA["Cross-Attention"]
            LN4["Layer Norm"]
            Add4["+"]
            FFN2["Feed-Forward<br/>Network"]
            LN5["Layer Norm"]
            Add5["+"]
            Dropout3["Dropout"]
            Dropout4["Dropout"]
            Dropout5["Dropout"]
            
            MHA2 --> Dropout3
            Dropout3 --> Add3
            LN3 --> Add3
            Add3 --> CMHA
            CMHA --> Dropout4
            Dropout4 --> Add4
            LN4 --> Add4
            Add4 --> FFN2
            FFN2 --> Dropout5
            Dropout5 --> Add5
            LN5 --> Add5
        end
        
        DecBlock --> DecOut["Decoder Output"]
    end
    
    subgraph Output["📤 OUTPUT"]
        Linear["Linear Layer"]
        SoftMax["Softmax"]
        Probs["Probability Distribution"]
        
        Linear --> SoftMax
        SoftMax --> Probs
    end
    
    EncOut --> DecBlock
    DecOut --> Linear
    
    style Encoder fill:#bbdefb
    style Decoder fill:#ffe0b2
    style Output fill:#c8e6c9
    style EncDetails fill:#e3f2fd
    style DecDetails fill:#fff3e0
```

---

## 🔍 Multi-Head Attention Mechanism

```mermaid
graph LR
    Q["Query"] --> W_Q["W_Q"]
    K["Key"] --> W_K["W_K"]
    V["Value"] --> W_V["W_V"]
    
    W_Q --> SplitQ["Split into<br/>h heads"]
    W_K --> SplitK["Split into<br/>h heads"]
    W_V --> SplitV["Split into<br/>h heads"]
    
    SplitQ --> Attn["Attention(Q,K,V)"]
    SplitK --> Attn
    SplitV --> Attn
    
    Attn --> Concat["Concatenate<br/>heads"]
    Concat --> W_O["W_O"]
    W_O --> Output["Output"]
    
    style Attn fill:#fff9c4
    style Output fill:#c8e6c9
```

---

## 💻 Implementation Guide

<details>
<summary><b>Step-by-Step Implementation</b></summary>

### Step 1: Positional Encoding
```python
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
```

### Step 2: Multi-Head Attention Class
```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        # Linear transformations for Q, K, V
        # Scaled dot-product attention
        # Output projection
```

### Step 3: Transformer Block
```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, dff)
        # Layer normalizations and dropouts
```

### Step 4: Encoder & Decoder
```python
class Encoder(tf.keras.layers.Layer):
    # Stack of transformer blocks
    # Embedding + Positional Encoding
    
class Decoder(tf.keras.layers.Layer):
    # Stack of transformer blocks with cross-attention
    # Embedding + Positional Encoding
```

### Step 5: Full Transformer Model
```python
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, ...):
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.final_layer = Dense(target_vocab_size)
```

</details>

---

## 🚀 Usage Example

```python
# Initialize model parameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 8500
target_vocab_size = 8000
maximum_position_encoding = 10000

# Create transformer model
transformer = Transformer(
    num_layers, d_model, num_heads, dff,
    input_vocab_size, target_vocab_size,
    maximum_position_encoding
)

# Prepare input data
inputs = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=input_vocab_size)
targets = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=target_vocab_size)

# Forward pass
output = transformer((inputs, targets), training=True)
print(f"Output shape: {output.shape}")  # (64, 50, 8000)
```

**Output Interpretation:**
- **64**: Batch size (64 sentences)
- **50**: Sequence length (50 tokens per sentence)
- **8000**: Target vocabulary size (probability distribution over 8000 tokens)

---

## 🧠 Key Concepts

<details>
<summary><b>Important Concepts Explained</b></summary>

### Attention Mechanism
Attention = Softmax(Q·K^T / √d_k) · V

Where:
- **Q (Query)**: What are we looking for?
- **K (Key)**: What can we find?
- **V (Value)**: What do we return?

### Self-Attention vs Cross-Attention
- **Self-Attention**: Q, K, V come from the same input
- **Cross-Attention**: Q from decoder, K, V from encoder output

### Masking
- **Padding Mask**: Prevents attention to padding tokens
- **Look-Ahead Mask**: Prevents decoder from seeing future tokens (autoregressive)

### Scaling Factor
√d_k prevents softmax from having extreme gradients when d_k is large

### Residual Connections
Enable training deeper networks by creating shortcuts: `Output = Input + f(Input)`

### Layer Normalization
Normalizes across feature dimension: `LayerNorm(x) = γ · (x - μ) / σ + β`

</details>

---

## 📁 Project Structure

```
Transformer-Model-from-Scratch/
├── README.md                 # This file
├── Transformer.ipynb        # Complete implementation notebook
│   ├── Imports
│   ├── Positional Encoding
│   ├── Multi-Head Attention
│   ├── Feed-Forward Network
│   ├── Transformer Block
│   ├── Encoder
│   ├── Decoder
│   ├── Full Transformer Model
│   └── Training & Testing
└── requirements.txt         # Dependencies
```

---

## 📊 Model Parameters Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_layers` | 4 | Number of encoder/decoder layers |
| `d_model` | 128 | Embedding dimension |
| `num_heads` | 8 | Number of attention heads |
| `dff` | 512 | Feed-forward hidden dimension |
| `input_vocab_size` | 8500 | Input vocabulary size |
| `target_vocab_size` | 8000 | Target vocabulary size |
| `dropout_rate` | 0.1 | Dropout probability |
| `max_position_encoding` | 10000 | Maximum sequence length |

---

## 🎓 Learning Resources

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual Guide
- [TensorFlow Documentation](https://www.tensorflow.org/guide/keras)

---

## 📝 Notes

- The model uses **positional encoding** instead of RNN recurrence to capture sequence position
- **Multi-head attention** allows the model to focus on different representation subspaces
- **Residual connections** and **layer normalization** stabilize training
- The decoder uses **causal/look-ahead masking** to prevent attending to future tokens

---

## 📄 License

This project is licensed under the MIT License.

---

**Last Updated**: April 2026 | **Status**: ✅ Complete & Functional
