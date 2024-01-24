## Inspired from https://towardsdatascience.com/pretrained-word-embeddings-using-spacy-and-keras-textvectorization-ef75ecd56360

### Trying Martino Mensio "Universal Sentence Encoder"
import spacy_universal_sentence_encoder


nlp = spacy_universal_sentence_encoder.load_model('en_use_md')

# your sentences
search_doc = nlp("This was very strange argument between american and british person")

main_doc = nlp("He was from Japan, but a true English gentleman in my eyes, and another one of the reasons as to why I liked going to school.")

print(main_doc.similarity(search_doc))
# this will print 0.310783598221594


phrase1 = nlp('I like to work in my office.')
phrase2 = nlp('I hate to go to my office.')

print(phrase1.similarity(phrase2))