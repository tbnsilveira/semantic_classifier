import spacy
import spacy_universal_sentence_encoder
import numpy as np
from keras.models import load_model
import pickle

class TextClassifier:
    def __init__(self, model_multiclass_path, encoder_multiclass_path, model_regression_path):
        # Carregar os modelos SpaCy
        self.nlp_core = spacy.load('en_core_web_lg')
        self.nlp_use = spacy_universal_sentence_encoder.load_model('en_use_lg')
        
        # Carregar os modelos de classificação e encoders
        self.model_multiclass = load_model(model_multiclass_path)
        with open(encoder_multiclass_path, 'rb') as f:
            self.encoder_multiclass = pickle.load(f)
        
        self.model_regression = load_model(model_regression_path)


    def nlp_getVector(self, word):
        '''
        Obtains the vector representation of a given word from SpaCy word embedding 'en_use_lg'.
        Usage:  nlp_getVector(word)[0] to get the text; nlp_getVector(word)[1] to get the vector.
                var_text, var_vector = nlp_getVector(word)
        '''
        try:
            word_vector = self.nlp_use(word).vector
            word_text = self.nlp_use(word).text
        except:
            print('Error: word vector not available.')
            return None
        return (word_text, word_vector)


    def similarity_findWords(self, keyword, n=10):
        # Checks if keyword is in the vocabulary
        if keyword not in self.nlp_core.vocab.strings:
            return []
        
        # Obtains the keyword vector representation
        keyword_vector = self.nlp_core.vocab[keyword].vector
        if keyword_vector is None or np.all(keyword_vector == 0):
            return []

        similarities = []

        for word_string in self.nlp_core.vocab.strings:
            word = self.nlp_core.vocab[word_string]
            if word.has_vector and not word.is_stop and word.is_alpha:
                similarity = np.dot(keyword_vector, word.vector) / (np.linalg.norm(keyword_vector) * np.linalg.norm(word.vector))
                similarities.append((word.text, similarity))
        
        # Sorting related words by similarity
        similarities = sorted(similarities, key=lambda item: item[1], reverse=True)

        # Return top n-words most similar, except the keyword.
        return [item for item in similarities[1:n+1] if item[0].lower() != keyword.lower()]


    def similarity_compareSentences(self, input_text, reference_texts, n=5):
        """
        Finds the n closest embeddings in the reference_texts to the input_text.
        
        Parameters:
            input_text (str): The word or sentence to find closest embeddings for.
            reference_texts (list of str): A list of reference texts to compare against.
            n (int): Number of closest embeddings to find.

        Returns:
            list of tuples: Each tuple contains the reference text and its cosine similarity to the input_text.
        """
        # Compute the vector for the input text
        input_vector = self.nlp_use(input_text).vector
        
        # Placeholder for storing similarities
        similarities = []
        
        for text in reference_texts:
            # Compute the vector for each reference text
            text_vector = self.nlp_use(text).vector
            # Calculate cosine similarity
            cosine_similarity = np.dot(input_vector, text_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(text_vector))
            # Append the similarity and reference text to the list
            similarities.append((text, cosine_similarity))
        
        # Sort the list of tuples by similarity in descending order
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return the top n most similar reference texts
        return similarities[:n]

    
    def textClassifier_multiclass(self, text):
        '''Given a word vector, classifies its content as perceptual, manifest, contextual, or latent, according to SCA.
        Usage example: wordClassifier_multiclass(word='trial', model=model_02_C, encoder=encoder_oneHot_C).
        '''
        new_entry = self.nlp_getVector(text)
        if new_entry:
            vector = self.nlp_getVector(text)[1]
            vector = np.expand_dims(vector, axis=0)
            result = self.model_multiclass.predict(vector)
            decoded_result = self.encoder_multiclass.inverse_transform(result)
        else:
            print('Word not existent in database.')
            return
        print(f'--- SCA: "{text}" has {decoded_result[0][0]} content.')
        print(f'Model used: {self.model_multiclass.name}')  # Print the name of the model
        return
        

    def textClassifier_regression(self, word):
        '''Given a word vector, shows the probability for objective and subjective semantic content, respectively.'''
        new_entry = self.nlp_getVector(word)
        if new_entry:
            vector = self.nlp_getVector(word)[1]
            vector = np.expand_dims(vector, axis=0)
            result = self.model_regression.predict(vector)
        else:
            print('Word not existent in database.')
            return
        print(f'--- {word}:\n{result[0][0]*100:.2f} of objectivity\n{result[0][1]*100:.2f} of subjectivity')
        #print(f'Model used: {self.model_regression.name}')  # Print the name of the model
        return
