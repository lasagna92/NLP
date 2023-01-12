import spacy
nlp = spacy.load('en_core_web_md')

# Example 1
word1 = nlp('cat')
word2 = nlp('monkey')
word3 = nlp('banana')

print(word1.similarity(word2))

word1 = nlp('Donald Trump')
word2 = nlp('Hillary Clinton')
word3 = nlp('Barack Obama')

print(word1.similarity(word2))

# I found extremely interesting how the model gives more than 50% of similarity because it recognized the relationship between monkey and banana 
# Following my example, is stunning how the model relate the three politician and gives more similarity that the example above

# Example 2
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Example 3
sentence_to_compare ="Why is my cat on the car"
sentences = ["where did my dog go",
            "Hello, there is my car",
            "I\'ve lost my car in my car",
            "I\'d like my boat back",
            "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# Running the example file with the small model you can notice straight away that the similarity index drops a lot and that's why the model doesn't use word vectors 
# Accordingly the printed result has a less accurate similarity index.