## Fraud detection using graph analytics

### Introduction
Graph analytics can be used to obtain more context characterized by a network of interconnected transactions between accounts or parties. This has emerged at the forefront of a new technology to support anti-money laundering analysis because money laundering involves transaction-flow relationships between various entities (in the form of a network).

The use of graph analytics can complement traditional machine learning techniques by overcoming the challenge of uncovering relationships in a massive, complex and interconnected dataset. Another advantage of using graph based methods is that the method is channel/platform independent and they are more adaptive than rule-based systems, and more expressive than traditional statistical methods. According to Interpol, most financial crimes are cross-border, or trans-national in nature. Graph analytics will provide the information needed to understand interconnectivity-based transactions at a wider or global scale. 

**Aim**
<br> Thus, the aim of this project is to explore application of network analytics and graph theory for fraud detection. 

### Methods
**Data**
<br> Data is transactional information from the bank to other accounts within the bank, and other target banks from 1. Jan 2020 to 31. May 2021. Features used in this study are source account holder, target account holder, date of transaction, transaction amount, and label (fraud/not fraud).  

**Levels of transactions**
<br> We are interested in transactions up to and including three degrees of connections.
<br> Level 1 – first degree connections
<br> Since the dataset is huge, the focus at this level is transactions to pay or within the bank. 

Level 2 – second degree connections
<br> We wanted to examine connections up to and including the second degree outside of the bank network, ie., transaction is first performed within bank (first degree), and then transaction is performed to an account outside of bank (second degree). To prevent the graph from increasing exponentially, the second transaction, performed by the second degree of connection has to be performed by the next day. For example, if the first transaction was performed on the 3rd March, the second transaction has to be performed by the 4th March.

Level 3 – third degree connections
<br> We wanted to examine connections up to and including the third degree outside of the bank network, ie., transaction is first performed within bank (first degree). Then, that target account performed a transaction to another account within bank (second degree), and further on to an account outside of bank (third degree). To prevent the graph from increasing exponentially, the next transaction, performed by the second/third degree of connection has to be performed by the next day. For example, if the first transaction was performed on the 3rd March, the second transaction has to be performed by the 4th March, and the third transaction has to be performed by the 5th March. 

**Centrality measures** 
<br> Five centrality measures are employed to represent the relative location and importance of an account. They are degree centrality, betweenness centrality, closeness centrality, eigenvector centrality and pagerank.



|   | notebook                      | description                    |
|---|-------------------------------|--------------------------------|
| 1 | --  | extracting levels of transactions |
| 2 | --  | EDA of a subset of transactions within 1st degree of connection |
| 2 | [EDA_level_2](https://github.com/doscsy12/ADI_projects/blob/main/AML/EDA_level_2.ipynb)| EDA of transactions at 2nd degree of connection |
| 2 | [EDA_level_3](https://github.com/doscsy12/ADI_projects/blob/main/AML/EDA_level_3.ipynb)| EDA of transactions at 3rd degree of connection |

