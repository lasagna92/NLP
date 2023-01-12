import sys
import spacy
import numpy

# Load the medium spacy model 
nlp = spacy.load('en_core_web_md')
# Create two empty list, one for the movies descriptions and the other one for the titles
movie_list = []
title_list=[]
# Open the file movies.txt and append to the above lists the data
try:
    with open('movies.txt', 'r') as f:  # Read the file inventory.txt
            # Loop through the lines of the file
        for line in f:
            split_line = line.strip().split(':')
            movie_list.append(split_line[1])
            title_list.append(split_line[0])
                 
except FileNotFoundError:
    print('\nFile not found!')
    sys.exit()

# Create a variable of the hulk movie description and convert it to NLP
hulk_movie = """Will he save their world or destroy it? When the Hulk becomes too dangerous for the
Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a
planet where the Hulk can live in peace. Unfortunately, Hulk land on the
planet Sakaar where he is sold into slavery and trained as a gladiator."""
hulk = nlp(hulk_movie)

similarity_index = []
# Function that takes a description of a movie as parameter and return the most similar movie in the file movies.txt
def next_movie(description):
    # Append the similarity index to the above list
    for movie in nlp.pipe(movie_list):
        similarity_index.append(description.similarity(movie))
         
    # Get the highest similarity index and return the movie's title
    id_max = numpy.argmax(similarity_index)
    return title_list[id_max]

# Call the function
print(f'The next movie to watch is: {next_movie(hulk)}')
    
    




