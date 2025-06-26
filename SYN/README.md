## Synthetic data generation (SYN)

### Introduction
There are many techniques utilised for synthetic data generation, and we need to have a thorough understanding of their effectiveness, the assessment of statistical similarities, and the maintenance of data integrity. Central to performing a systematic review is the examination of the robustness and reliability of the algorithms utilized. For example, Generative Adversarial Networks (GANs), while known for their potential, are notoriously hard to train, and susceptible to instability issues such as divergence, which pose risks to the authenticity of their outputs. Furthermore, evaluating the fidelity of generated samples remains an ongoing challenge, as conventional metrics may inadequately capture certain characteristics present in real datasets.  

### Aim 
So, the aim of performing a systematic review is to navigate through these complexities, and to ensure a comprehensive understanding of the efficacy and limitations of synthetic data generation techniques in the financial domain.

#### Systematic review 
A systematic literature review was conducted to examine existing research on synthetic data generation techniques within the financial domain. This method includes defining the research, defining article search method, and analysis of the results after full text screening of selected articles. The results of this process will inform the development of our algorithms aimed at accurately and reliably replicating financial transactions.  

#### Evaluating Generative Models for Synthetic Financial Transaction Data
The banking sector faces challenges in using deep learning due to data sensitivity and regulatory constraints, but generative AI may offer a solution. Thus, this study identifies effective algorithms for generating synthetic financial transaction data and evaluates five leading models - Conditional Tabular Generative Adversarial Networks (CTGAN), DoppelGANger (DGAN), Wasserstein GAN, Financial Diffusion (FinDiff), and Tabular Variational AutoEncoders (TVAE) - across five criteria: fidelity, synthesis quality, efficiency, privacy, and graph structure. While none of the algorithms is able to replicate the real data's graph structure, each excels in specific areas: DGAN is ideal for privacy-sensitive tasks, FinDiff and TVAE excel in data replication and augmentation, and CTGAN achieves a balance across all five criteria, making it suitable for general applications with moderate privacy concerns. As a result, our findings help guide decision-makers in choosing the right generative AI model to support safe, effective, and scalable use of synthetic data in financial services.

#### Algorithmic Development and Evaluation for Replicating Temporal and Graph Patterns in Financial Data
This topic is important because financial data often exhibits complex temporal patterns and network structures, such as transactions over time and relationships between accounts. Accurately replicating these aspects allows for the creation of realistic datasets that reflect the dynamic and interconnected nature of financial systems. This supports the development, testing, and benchmarking of AI models, such as those used in fraud detection, credit scoring, and risk assessment—under realistic conditions. By capturing both the time-based behavior and relational structures, these algorithms help improve model performance, robustness, and relevance in real-world financial applications.


|   | file                          | description                    |
|---|-------------------------------|--------------------------------|
|1. | Algorithm development | Algorithm development |
|2. | [Article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5183254) | Accepted article in WITS2024 |
|3. | [Podcast on paper](https://soundcloud.com/sook-yee-chong/synthetic-data-notebooklm-podcast?si=0493373339ec4376ba707f8afa29e789&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)  | Podcast created by NotebookLM |
|4. | [Friday Hacks](https://www.youtube.com/watch?v=Q2ct1z-e5pM)  | Talk to NUS students |
|5. | Temporal Graph algorithm | Algorithm development |