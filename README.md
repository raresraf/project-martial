# Project Martial

The core part of our research is focused around detecting plagiarism. While there were many cases of _plagiarism_ in the history, the first documented case of using the word _plagiarius_ was during the life of the roman poet, Martial. He is considered to be the first person to claim authorship rights, in an ancient world where intellectual property and copyright lays were not enforced, when exact copies of his poems and epigrams started to appear presented as personal work by obscure writers. Project Martial is a initiative aiming to provide automatic assistance in detecting software plagiarism.

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

![](martial-ui-rcomplexity.png?raw=true)

### On Plagiarism and Software Plagiarism

TBD

https://link.springer.com/book/9783031702587
