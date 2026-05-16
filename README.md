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

## Acknowledgements

This repository uses the Atten-Mixer mechanism proposed by Zhang et al. (WSDM 2023). In their [1] paper, Zhang et al. proposed to remove the GNN propagation part and allow the readout module to take more responsibility in the model reasoning process. They introduced the Multi-Level Attention Mixture Network (Atten-Mixer), which leverages both concept-view and instance-view readouts for multi-level reasoning over item transitions.

[1] Peiyan Zhang, Jiayan Guo, Chaozhuo Li, Yueqi Xie, Jaeboum Kim, Yan Zhang, Xing Xie, Haohan Wang, Sunghun Kim. Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network. WSDM 2023: 168-176.

Original Atten-Mixer paper: https://doi.org/10.1145/3539597.3570445  
Original Atten-Mixer code and data: https://github.com/Peiyance/Atten-Mixer-torch
