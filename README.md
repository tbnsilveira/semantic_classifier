# Semantic Content Classifier

Welcome to the Semantic Content Classifier project, where we explore the nuances of language through **objective** and **subjective** categorizations of words, thus extended to sentences and texts. This project centers around the SCA_Utils library, a tool developed to enhance text analysis by leveraging pre-trained SpaCy embeddings and Semantic Content Analysis (SCA).


## Technical Overview
At the core of our Semantic Similarity project is the SpaCy Universal Sentence Encoder (USE), a powerful tool designed to convert text into high-dimensional vector representations. These vectors capture the semantic essence of words and sentences, enabling nuanced language understanding. We implemented the main SCA and USE methods in the library [sca_utils.py](https://github.com/tbnsilveira/semantic_similarity/blob/main/source/sca_utils.py).


The SCA_Utils library provides tools for analyzing text for semantic similarity, focusing on objective and subjective categorizations using SpaCy's word embeddings. It makes use of pre-trained models available in [this directory](https://github.com/tbnsilveira/semantic_similarity/tree/main/models).

### Features:

- Categorizes words and sentences as objective or subjective.
- Applies SCA to SpaCy's word embeddings for enhanced semantic analysis.
- Assesses semantic similarity between words or sentences.

### Quick Start:

1. Install SpaCy and download the necessary language models:

    ```bash
    pip install spacy
    python -m spacy download en_core_web_lg
    python -m spacy download en_use_lg
    ```

2. Import the `TextClassifier` from the library:

    ```python
    import sys
    sys.path.append("../source")     # Add the directory 'source' to sys.path

    from sca_utils import TextClassifier
    ```

3. Analyze a sentence:

    ```python
    classifier = TextClassifier(model_multiclass_path='../models/model_02_E.h5',
                                encoder_multiclass_path='../models/encoder_oneHot_E.pickle',
                                model_regression_path='../models/model_01_D2.h5')
    print(classifier.nlp_getVector("Your sample sentence"))
    ```

4. Estimate the objective and subjective portions of a sentence:

    ```python
    classifier.textClassifier_regression('Your sample sentence')
    ```

    Results in:
    > --- Your sample sentence:  
    > 38.72 of objectivity  
    > 15.49 of subjectivity  


## License
This project is licensed under the MIT License. The MIT License is a permissive free software license that allows for private, commercial, and public use, provided the original license and copyright notice are included with any substantial portions of the software. This makes the MIT License suitable for both open-source and commercial projects.


## Acknowledgments
This project builds upon the powerful NLP capabilities provided by SpaCy and its word embeddings, enabling a deeper dive into the semantic relationships between words. 

This project is part of the Ph.D. thesis of Tiago B. N. Silveira.  
UTFPR, Brazil. 2023-12.  

## Cite this repository
If you use this software in your work, please cite it using the following metadata:

Da Silveira, T. B. N. (2023). Semantic Similarity (Version 1.0.0) [Computer software]
