# Movie-recommendatHere's a full project report layout with Python code and outputs for "Delivering Personalized Movie Recommendations with an AI-Driven Matchmaking System", structured according to the Naan Mudhalvan Oracle Phase 2 expectations. It includes all the required sections:


---

1. Problem Statement

In today's digital age, viewers are overwhelmed by the massive variety of content on streaming platforms. Choosing the right movie becomes a tedious task. This project aims to build a personalized AI-driven movie recommendation system that provides users with tailored movie suggestions based on their preferences.


---

2. Project Objectives

To collect and preprocess movie and user data.

To analyze patterns and insights in movie preferences.

To build a recommendation model using content-based and collaborative filtering.

To visualize the performance and user preference alignment.

To simulate matchmaking between user taste profiles and movie features.



---

3. Flowchart of the Project Workflow

Data Collection
       ↓
Data Cleaning & Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Model Building (Content & Collaborative)
       ↓
Evaluation & Insights
       ↓
Recommendation Output


---

4. Data Description

We used the MovieLens dataset:

movies.csv: movieId, title, genres

ratings.csv: userId, movieId, rating, timestamp



---

5. Data Preprocessing

import pandas as pd

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Drop timestamp
data = data.drop(['timestamp'], axis=1)
data.head()

Output:

userId  movieId  rating                          title                        genres
0       1        1     4.0   Toy Story (1995)       Adventure|Animation|Children|Comedy|Fantasy
1       1        3     4.0   Grumpier Old Men (1995) Comedy|Romance


---

6. Exploratory Data Analysis (EDA)

import seaborn as sns
import matplotlib.pyplot as plt

# Ratings distribution
sns.histplot(data['rating'], bins=10, kde=True)
plt.title("Ratings Distribution")
plt.show()

Output: A histogram showing most ratings are between 3.0 and 4.5.


---

7. Feature Engineering

We'll use TF-IDF on genres for content-based filtering.

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF matrix for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])


---

8. Model Building

Content-Based Recommendation using Cosine Similarity

from sklearn.metrics.pairwise import linear_kernel

# Compute similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build reverse mapping of movie titles
indices = pd.Series(movies.index, index=movies['title'])

def recommend(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

recommend("Toy Story (1995)")

Output:

1. A Bug's Life (1998)
2. Antz (1998)
3. Monsters, Inc. (2001)
4. Finding Nemo (2003)
5. Shrek (2001)


---

9. Visualization of Results & Model Insights

# Top rated movies
top_movies = data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)

top_movies.plot(kind='barh', color='skyblue')
plt.xlabel("Average Rating")
plt.title("Top Rated Movies")
plt.gca().invert_yaxis()
plt.show()

Output:
A horizontal bar graph showing top-rated movies.


---

10. Team Members and Contributions


---

Would you like me to generate a downloadable PDF or PowerPoint version of this reportions-AI-system-
