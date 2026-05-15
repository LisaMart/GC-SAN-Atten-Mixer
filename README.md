**GC-SAN model integrated with Atten-Mixer (Atten-Mixer Enhanced Graph Neural Network for Session-based Recommendation).**
*Atten-Mixer mechanism originally proposed by Zhang et al., [WSDM 2023]*

📌 **Problem**
State-of-the-art session-based recommender systems (SBRS) using Graph Neural Networks (GNN) face two main challenges:

* High computational cost due to multi-layer message passing.
* Difficulty capturing both short-term session dynamics and global session intent, especially under sparse or noisy session data.

✅ **Solution**
The Atten-Mixer module (Zhang et al., WSDM 2023) replaces heavy GNN propagation with a lightweight, multi-level attention-based readout. 

📊 **Results**

* Up to 99% reduction in training time compared to original GC-SAN
* Maintains competitive recommendation performance

**Metrics:** Precision@N, MRR@N
**Datasets:** Diginetica, Yoochoose1_64
