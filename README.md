GC-SAN model combined with AMGNN-SR (Atten-Mixer Enhanced Graph Neural Network for Session-based Recommendation). 

📌 Problem
Traditional Graph Neural Networks (GNNs) for session-based next-item recommendation suffer from **high computational cost** due to deep message-passing layers, while still struggling to **balance short-term dynamics and long-term intent**, especially under **sparse or noisy session data**.

✅ Solution
AMGNN-SR replaces heavy GNN propagation with a **lightweight Atten-Mixer module** — a multi-level attention-based readout that aggregates session semantics in **linear time**.  
- No multi-hop propagation  
- One-shot attention pooling  
- Captures both local transitions and global intent

📊 Results
- **99% reduction in training time** vs. GC-SAN on Diginetica  
- **Precision@20 improved by +0.5%**  

Metrics - MRR@K, Precision@K
Datasets - diginetica, yoochoose1_64
