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
<<<<<<< HEAD
-[x]  Prepare interaction matrix using dataset.
-[X]  Implemented classical models.
-[x]  Understand and implement user-based k-Nearest Neighbors (k-NN) for collaborative filtering.
-[x]  Evaluated models using appropriate metrics.
-[x]  Compared model performances.
=======
- [x]  Prepare interaction matrix using dataset.
- [ ]  Implemented classical models.
- [ ]  Understand and implement user-based k-Nearest Neighbors (k-NN) for collaborative filtering.
- [ ]  Evaluated models using appropriate metrics.
- [ ]  Compared model performances.
>>>>>>> ff3e38f136324e4a7bb4301c29e96e1f13c0fe56

### What I Learned
- Understood how k-Nearest Neighbors (k-NN) and logistic regression can be applied in recommender systems.
- Gained familiarity with Precision@k and Recall@k as evaluation metrics for top-k recommendation quality.
- Learned to split data appropriately for model evaluation (train-test split, cross-validation).
- Compared model performances quantitatively and identified strengths/limitations of classical approaches.
- Realized the interpretability and simplicity of classical models as baselines for more complex techniques.

#### Dataset
- **Source:** Provided `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).
- Used only userId, movieId, and rating for collaborative filtering.
- Data is split per user into training and test sets.
<hr>
⚠️ Note : I have not yet not finished with my week 4 as i am still stuck between k-nn model and liogistic regression  model 
- Trying to figure out !! 

## Week 5

### ✅Goals
-[x]   Developed hybrid recommendation model.
-[x]  Addressed cold-start problem.
-[x]  Created visualizations of embeddings.

### What I Learned
- Explored the concept of hybrid recommender systems by combining content-based and collaborative filtering.
- Understood how to leverage user and item metadata (like genres, age, etc.) to mitigate the cold-start problem.
- Implemented strategies that blend multiple recommendation sources for better accuracy and coverage.
- Visualized high-dimensional embeddings using PCA and t-SNE to inspect similarities between users and items.
- Learned that hybrid models offer flexibility and can be fine-tuned to handle different data scenarios.

## Week 6

### ✅Goals
-[x]  Completed study of neural networks.
-[x]  Implemented deep learning recommendation model.
-[x]  Compared model performances.

### What I Learned
- Studied the architecture of neural networks, focusing on embedding layers for user/item representation.
- Implemented a neural collaborative filtering (NCF) model using Keras.
- Observed how deep learning models can capture complex, non-linear interactions between users and items.
- Compared the deep learning model’s performance with classical models using the same evaluation metrics.
- Gained insight into scalability and generalization benefits of deep models in large-scale recommendation systems.

## Week 7

### ✅Goals
-[x]  Developed frontend interface.
-[x]  Integrated APIs with frontend.
-[x]  Implemented user interaction features.

### What I Learned
- Built a user-friendly frontend for the recommendation system using Streamlit.
- Created and tested RESTful APIs to serve model predictions dynamically.
- Integrated backend models with the frontend to allow real-time recommendation retrieval.
- Designed a basic user interaction system (e.g., feedback, likes/dislikes) for future personalization.
- Understood how to structure a full-stack ML application with clear separation between frontend and backend.

---

INTELLIREC is an end-to-end movie recommendation system developed in an 8-week learning sprint. It integrates classical recommendation algorithms, deep learning models, hybrid systems, and a full-stack application interface with planned deployment.

---

## 📁 Folder Structure with File Descriptions

### • Week 1: Content-Based Filtering
- `cosine_similarity.py` • Implements cosine similarity for content-based filtering.
- `data_preprocess.py` • Preprocesses raw movie dataset for modeling.
- `recommendation_engine.py` • Runs the content-based recommendation engine.
- `testdataset.csv` • Sample input data for testing similarity logic.

### • Week 2: Probability & Exploratory Data Analysis
- `clean_merge_dataset.py` • Merges and cleans multiple movie datasets.
- `eda.py` • Performs EDA with statistical analysis and plots.
- `cleaned_merged_movies.csv` • Final merged and cleaned dataset.
- `movies.csv` • Raw movies data (titles, genres, etc).
- `ratings.csv` • Raw user ratings for movies.
- `*.png` • Visualizations: rating distributions, genre stats, etc.

### • Week 3: Collaborative Filtering with SVD
- `svd_recommendation.py` • SVD-based movie recommendation for all users.
- `svd_recommendation_user.py` • Generates personalized recommendations using SVD.
- `ratings.csv` • Ratings data used for matrix factorization.

### • Week 4: Classical ML Models & Evaluation
- `k-nn.py` • k-Nearest Neighbors model for movie recommendation.
- `logistic_regression.py` • Logistic regression model for prediction-based recommendations.
- `compare_models.py` • Script to compare classical model performance.
- `evaluation_metrics.py` • Calculates Precision@k and Recall@k for evaluation.
- `interaction_matrix.py` • Builds user-item interaction matrix.
- `cleaned_merged_movies.csv` • Cleaned movie dataset reused here.
- `movies.csv` • Movies metadata.
- `ratings.csv` • Ratings data.

### • Week 5: Hybrid Recommenders & Cold Start Problem
- `hybrid_model.py` • Combines content and collaborative filtering methods.
- `cold_start_solution.py` • Handles cold-start problem using item metadata.
- `visualize_embeddings.py` • Generates PCA and t-SNE plots for embeddings.
- `movie_pca.png`, `movie_tsne.png` • Movie embedding visualizations.
- `user_pca.png`, `user_tsne.png` • User embedding visualizations.
- `movies.csv` • Movies metadata.
- `ratings.csv` • Ratings data.

### • Week 6: Deep Learning for Recommendations
- `deep_model.py` • Builds and trains a neural collaborative filtering model.
- `compare_with_classical.py` • Compares deep learning model against classical ones.
- `movie_recommender.py.py` • Deep learning-based movie recommender script.
- `deeplearning_model.h5` • Trained deep learning model (Keras format).
- `utils.py` • Helper functions for deep learning and preprocessing.
- `movies.csv` • Movie data used for training DL model.
- `data/` • Subdirectory for model-specific inputs.

### • Week 7: Frontend Development & API Integration
- `app.py` • Streamlit or Flask-based frontend app script.
- `api.py` • Backend API for serving model recommendations.
- `recommender.py` • Interface that links model logic to API endpoints.
- `utils.py` • Shared utility functions for API and frontend.
- `rating.txt` • Placeholder or test data for frontend interaction.
- `models/` • Folder to store serialized models.
- `data/` • Folder for frontend-compatible data.

### • Week 8: Deployment & Documentation
- **[Planned]** • Deployment setup using Heroku, Vercel, or GCP.

---

## 🚀 Project Highlights

- ✅ Content-Based Filtering (Cosine Similarity)
- ✅ Collaborative Filtering with SVD
- ✅ Classical ML Models (k-NN, Logistic Regression)
- ✅ Hybrid Recommenders with Cold-Start Handling
- ✅ Deep Learning with Neural Collaborative Filtering
- ✅ Embedding Visualizations (PCA, t-SNE)
- ✅ API & Frontend (Flask/Streamlit)
- ✅ Final Deployment Plan

---

Developed with 💡 by **Vaibhav**

---

⭐ If you find this helpful or inspiring, give it a ⭐
Thank You !!!
