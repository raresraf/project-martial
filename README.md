# Project Martial

The core part of our research is focused around code similarity detection and assisting human experts during plagiarism investigations. 

While there were many cases of _plagiarism_ in the history, the first documented case of using the word _plagiarius_ was during the life of the roman poet, Martial. He is considered to be the first person to claim authorship rights, in an ancient world where intellectual property and copyright lays were not enforced, when exact copies of his poems and epigrams started to appear presented as personal work by obscure writers. Project Martial is a initiative aiming to provide automatic assistance in detecting software plagiarism.

## Run project martial locally

### Backends

```bash
$ bazel run :main
```

### Frontend
```bash
$ cd ui
$ ng serve --open
```

## Techniques

### Code Comments: A Way Of Identifying Similarities In The Source Code

This research investigates if analyzing code comments can reveal functional similarities in software. The authors propose two novel models: one for machine-readable comments (e.g. linter directives) and another for human-readable comments (in natural language). 

The model for machine-readable comments utilizes multi-dimensional cosine distances to compute the similarity, based on one-hot encoded representations of the source code, while the model for human-readable comments employs natural language processing techniques, including Levenshtein distances, Word2Vec-based approaches, and deep learning models like sentence transformers. The research evaluates these models on some well known open-source projects, including VSCode and Kubernetes, and demonstrates the potential of code comments in detecting code similarities. This research is integrated into Project Martial. It has been published in the Journal of Mathematics, a peer-reviewed, open access journal published by MDPI.

https://www.mdpi.com/2227-7390/12/7/1073

![](martial-ui-comments.png?raw=true)

### Complexity Based Code Embeddings

This paper presents a method for converting algorithms into numerical representations (embeddings) by analyzing their runtime behavior with different inputs. The method uses profiling tools to collect runtime statistics, followed by fitting mathematical functions to these metrics in order to create the associated code embeddings.
The research demonstrates the effectiveness of these embeddings in classifying algorithms used in competitive programming challenges, achieving high precision and recall when applied on the task of classification. In the paper an XGBoost-based model is proposed. It has been published and presented in the Proceedings of the 15th International Conference on Computational Collective Intelligence and was awarded the Best Student Paper distinction.

https://link.springer.com/chapter/10.1007/978-3-031-41456-5_20

### Code Similarity Detection Using Complexity-Based Birthmarks

This research investigates using complexity-based code embeddings to produce unique software birthmarks optimized for software similarity detection. We suggest a novel approach to address this problem by handling the source code as a collection of complexity-based birthmarks. Our model has identified code implementation similarity across 280,000 analyzed pairs with an F1-score of 82% when evaluated against a dataset of solutions gathered from Codeforces competitive programming contests, using only twenty-four unique features for birthmark generation. Using a perf-based profiler that recorded raw metrics like branch misses, total CPU cycles, or page faults captured on a Unix system, the programs under evaluation were dynamically evaluated under a range of incremental inputs. The similarity score was subsequently determined using these performance metrics to build regression models designed to approximate the program’s complexity class for each performance metric.

https://link.springer.com/chapter/10.1007/978-3-032-09318-9_21

![](martial-ui-rcomplexity.png?raw=true)

### On Plagiarism and Software Plagiarism

This paper explores the complexities of automatic detection of software similarities, in relation of the unique challenges of digital artifacts and introduces Project Martial, an open-source software solution for detecting code similarity. This research enumerates some of the existing approaches to counter software plagiarism, by examining both the academia and legal landscape, including notable lawsuits and court rulings that have shaped the understanding of software copyrights infringements for commercial use applications. Furthermore, we categorize the classes of detection challenges, based on the available artifacts, and we provide a survey on the previously studied techniques in the literature, including solutions based on fingerprinting, software birthmarks or code embeddings and exemplify how a subset of them can be applied in the context of Project Martial.

https://link.springer.com/book/9783031702587

### Relational Database Similarity Detection via Network Traffic Analysis

In many real-world scenarios, analysts and security professionals encounter scenarios with limited to no information on the internal structure of software systems. In these circumstances, observable network traffic is essential in identifying software characteristics when source code and controlled execution are inaccessible. As such, we present a novel approach to network traffic pattern analysis for interpreting similarities between software applications when interactions are only possible from a client perspective. We leverage n-grams to identify distinct signatures in the application layer communication and exemplify their usage when applied to various relational database management systems (RDBMS). Our method involves constructing a corpus of network traffic produced by various database engines with different versions while operating everyday operations. We rely on TF-IDF scores to define corresponding numerical vectors for arbitrary sequences of communications. We show that 3-gram and 4-gram analysis can distinguish extremely well between database engines (i.e., MySQL, PostgreSQL, and SQL Server) even in scenarios when dealing with different major/minor versions of the same engines or different deployment environments (e.g., using public docker images deployed locally versus cloud-managed solutions).

https://ieeexplore.ieee.org/abstract/document/11181642/
