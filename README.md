# IntelliRec_SoC
<br>
This is my repository for my Seasons of Code Project : IntelliRec
<br>
IntelliRec : An Intelligent Recommendation Engine for Personalised User Experience<br>(ID:71)

## Week 1

### ✅Goals
- [x]  Completed study of linear algebra fundamentals.
- [x]  Implemented cosine similarity.
- [x]  Dataset collected and preprocessed.

### What I Knew Before
- Basics of Python and pandas
- Concepts of Linear Algebra (MA110)

### What I Learned
- How cosine similarity works and why it's used
- One-hot encoding and vector representation of text features
- Importance of data preprocessing in Machine Learning Projects
- Cleaning and preprocessing real-world data

## Week 2

### ✅Goals
- [x] Completed study of probability concepts.
- [x] Performed EDA on the dataset.
- [x] Visualizations created.

### What I Knew Before
- Basics of Probability and Statistics (IE102)
- Very light idea about Pandas Library

### What I Learned
- Visualized key patterns in movie ratings and genres using Matplotlib and Seaborn.
- Cleaned and prepared data for deeper analysis in future weeks.
- Simple plotting with matplotlib.
- Better Understanding of Pandas Library

## Week 3

### ✅Goals
- [x]  Completed study of SVD.
- [x]  Implemented collaborative filtering model.
- [x]  Evaluated model performance.

### What I Learned
🔸 Matrix Factorization (SVD)
SVD factorizes the user-item rating matrix ( R ) as:

[ R \approx U \Sigma V^T ]

Where:

( U ): User latent feature matrix
( \Sigma ): Importance of each latent feature
( V^T ): Item latent feature matrix
This helps estimate missing ratings by projecting users and items into a shared feature space.

🔸 RMSE Evaluation <br>
🔸 Implemented SVD using surprise library

## Week 4

### ✅Goals
- [x]  Prepare interaction matrix using dataset.
- [ ]  Implemented classical models.
- [ ]  Understand and implement user-based k-Nearest Neighbors (k-NN) for collaborative filtering.
- [ ]  Evaluated models using appropriate metrics.
- [ ]  Compared model performances.

### What I Learned

#### Dataset
- **Source:** Provided `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).
- Used only userId, movieId, and rating for collaborative filtering.
- Data is split per user into training and test sets.
<hr>
⚠️ Note : I have not yet not finished with my week 4 as i am still stuck between k-nn model and liogistic regression  model 
- Trying to figure out !! 

