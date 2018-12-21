
# Movie Recommendation with VAE and LightFM

Yijie Cao yc3544<br>
Xinxin Huang xh2389<br>
Chunran Yao cy2511<br>

## 1. Introduction 

### 1.1 GOAL

- The goal of a  good recommendation systems is showing users some unseen items they may like and helping users interact more effectively with items. Collaborative Filtering is a set of common methods in building recommenders. It makes predictions about the tastes of a user by collecting preferences information from lots of users. These sets of models assume that people like one item because they like other similar items, and people with similar tastes like similar items. The well-known Collaborative Filtering methods include memory-based approach (KNN) and model-based approach (SVD). In these cases, the recommender system literature focused on explicit feedback.

- However, as web grows bigger, the size of data become larger while these models are inherently linear, which limits their modeling capacity. Also, there is also a challenge where little data is available on new users and items. (Cold start problem) Moreover, for missing-not-at-random phenomenon: for example, he ratings that are missing are more likely to be negative precisely because the user chooses which his liked items to rate. In these cases, we are going to explore the implicit feedback from users. 

- In this project, we implemented variational autoencoders (vaes) to collaborative filtering and  a hybrid content-collaborative model, called LightFM for for implicit feedback.

### 1.2 DATA & Data preprocessing

**Original Data** :
We use MovieLens 20M dataset, it contains 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. 

 In the original data, there are 20000263 rating events from 138493 users and 26744 movies (sparsity: 0.540%)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>1112486027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>29</td>
      <td>3.5</td>
      <td>1112484676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>32</td>
      <td>3.5</td>
      <td>1112484819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>3.5</td>
      <td>1112484727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>3.5</td>
      <td>1112484580</td>
    </tr>
  </tbody>
</table>

**General Precedure** :

- We load the data and create train/test splits following strong generalization.

- For all the algorithm, we only keep items that are rated by at least 50 users. This method decrease the sparsity. 

- For FM, since it takes to long to train the model, we randomly sample the data, only using 10% of the whole dataset.
  - After filtering, there are 19847947 rating events from 138493 users and 10524 movies (sparsity: 1.362%) for lightfm.

- For VAE, we binarize the data by setting ratings >=4 equal to 1, and the others euqal 0.
  - After filtering, there are 9868061 rating events from 138287 users and 7345 movies (sparsity: 0.972%) for VAE.

## 2. Model definition

### 2.1  [Variational Autoencoders](https://arxiv.org/pdf/1802.05814.pdf)

__Notations__: We use $u \in \{1,\dots,U\}$ to index users and $i \in \{1,\dots,I\}$ to index items. In this work, we consider learning with implicit feedback. The user-by-item interaction matrix is the click matrix $\mathbf{X} \in \mathbb{N}^{U\times I}$. The lower case $\mathbf{x}_u =[X_{u1},\dots,X_{uI}]^\top \in \mathbb{N}^I$ is a bag-of-words vector with the number of clicks for each item from user u. We binarize the click matrix. It is straightforward to extend it to general count data.

__Generative process__: For each user $u$, the model starts by sampling a $K$-dimensional latent representation $\mathbf{z}_u$ from a standard Gaussian prior. The latent representation $\mathbf{z}_u$ is transformed via a non-linear function $f_\theta (\cdot) \in \mathbb{R}^I$ to produce a probability distribution over $I$ items $\pi (\mathbf{z}_u)$ from which the click history $\mathbf{x}_u$ is assumed to have been drawn:

$$
\mathbf{z}_u \sim \mathcal{N}(0, \mathbf{I}_K),  \pi(\mathbf{z}_u) \propto \exp\{f_\theta (\mathbf{z}_u\},\\
\mathbf{x}_u \sim \mathrm{Mult}(N_u, \pi(\mathbf{z}_u))
$$

- The objective for **$Multi-DAE$** for a single user $u$ is:
$$
\mathcal{L}_u(\theta, \phi) = \log p_\theta(\mathbf{x}_u | g_\phi(\mathbf{x}_u))
$$
where $g_\phi(\cdot)$ is the non-linear "encoder" function.


- The objective of **$Multi-VAE^{PR}$** (evidence lower-bound, or ELBO) for a single user $u$ is:
$$
\mathcal{L}_u(\theta, \phi) = \mathbb{E}_{q_\phi(z_u | x_u)}[\log p_\theta(x_u | z_u)] - \beta \cdot KL(q_\phi(z_u | x_u) \| p(z_u))
$$
where $q_\phi$ is the approximating variational distribution (inference model). $\beta$ is the additional annealing parameter that we control. The objective of the entire dataset is the average over all the users. 

###  2.2    [Factorization Machine](https://arxiv.org/pdf/1507.08439.pdf)

In LightFM, like in a collaborative filtering model, users and items are represented as latent vectors (embeddings).

Let $U$ be the set of users, $I$ be the set of items, $F^{U}$ be the set of user features, and $F^{I}$ the set of item features. Each user interacts with a number of items, either in a favourable way (a positive interaction), or in an unfavourable way (a negative interaction). The set of all user-item interaction pairs $(u, i) ∈ U × I $ is the union of both positive $S^+$ and negative interactions $S^−$.

Users and items are fully described by their features. Each user $u$ is described by a set of features $f_u ⊂ F^U$. The same holds for each item $i$ whose features are given by $f_i ⊂ F^I$ . The features are known in advance and represent user and item metadata.

The model is parameterised in terms of $d$-dimensional user and item feature embeddings $e^U_f$ and $e^I_f$ for each feature $f$. Each feature is also described by a scalar bias term ($b^U_f$ for user and $b^I_f$ for item features).

The latent representation of user $u$ and item $i$ are given by the sum of its features’ latent vectors:
$q_u=\sum\limits_{j \in f_U } e^U_j , p_i=\sum\limits_{j \in f_i } e^I_j $
The bias term for user $u$ and item $i$ are given by the sum of the features’biases:
$b_u =\sum\limits_{j \in f_U} b^U_j, b_i = \sum\limits_{j \in f_i} b^I_j $
The model’s prediction for user u and item i is then given by the dot product of user and item representations, ad- justed by user and item feature biases: $$\widehat{r}_{ui} =  f(q_u \cdot p_i + b_u+ b_i )$$

- The optimisation objective for the model consists in max- imising the likelihood of the data conditional on the param- eters. The likelihood is given by
$$ L(e^U, e^I, b^U, b^I)= \prod\limits_{(u,i) \in S^+} \widehat{r}_{ui} \times \prod\limits_{(u,i) \in S^-} (1-\widehat{r}_{ui}) $$

## 3. Results

### 3.1 KNN Result

Recall@20:  test 0.48.
Recall@50:  test 0.49.
Catalog coverage@20: test 0.05.
Catalog coverage@50: test 0.05.

### 3.2 SVD Result

Recall@20:  test 0.47.
Recall@50:  test 0.47.
Catalog coverage@20:  test 0.05.
Catalog coverage@50:  test 0.05.

### 3.3 LightFM Result (without side information)

Recall@20: train 0.33, test 0.12.
Recall@50: train 0.31, test 0.17.
NDCG@100: train 0.25, test 0.19.
CatalogCoverage@20: 0.18.
CatalogCoverage@50: 0.28.

### 3.4 VAE Result

- Without side information 

  NDCG at 100:  0.2596440767791029
  recall at 20:  0.4529321981995462
  recall at 50:  0.4866546361681343
  Catalog coverage@20: test 0.4543.
  Catalog coverage@50: test 0.5532.

- With side Information (genomes)

  NDCG at 100:  0.17503165510705818
  recall at 20:  0.29224579632398895
  recall at 50:  0.29288700258579703
  Catalog coverage@20: test 0.0050.
  Catalog coverage@50: test 0.0135.

### 3.5 Comparison and Criticism

| Models                  | recall@20 | recall@50 | ndcg@100    | coverage@20 | coverage@50 |
| ----------------------- | --------- | --------- | ----------- | ----------- | ----------- |
| KNN                     | 0.48      | 0.49      | ----------- | 0.05        | 0.05        |
| SVD                     | 0.47      | 0.47      | ----------- | 0.05        | 0.05        |
| FM                      | 0.12      | 0.17      | 0.19        | 0.18        | 0.28        |
| VAE (without side info) | 0.45      | 0.49      | 0.26        | 0.45        | 0.55        |
| VAE (add genomes)       | 0.29      | 0.29      | 0.18        | 0.005       | 0.0135      |



## 4. Conclusion

**Our work**



**limitation**

1. Our work is highly restricted by the computers due to our wrong estimation of the size of the dataset. Firstly, we have to sample the data and use a much smaller datset to train the models. Secondly, we failed to add genomes (side information) to FM,  because it fits model single-threadly , taking too much time to train and making our computers crash. Besides, in VAE we have to choose some small hyperparamters, specifically, the epoch and batch-size is set to a small value. We believe we can get better performance with more epochs and larger batch-size.
2. For the evaluation metrics, we just applied recall, ndcg and coverage. We didn't evaluate noverlty and serendipity, which is also very important in industry.
3. Due to time limit, we do not use IMDB data as side information. According to the [article](https://arxiv.org/pdf/1808.01006.pdf), adding IMDB data as side information in the VAE performs best among all the VAE approches. 

**Future work**

1. Using Spark: To better train model on this large dataset, it is necessary to train models with spark or other distributed software. 
2. Adding more side information: To better optimizing our model, we can add more side information, such as timestamp, IMDB and etc.
3. Hybrid model: By combining present models with content-based model,  we can make our model more robust. 

### *References*
[Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman and Tony Jebara (2018).](https://arxiv.org/pdf/1802.05814.pdf) Variational Autoencoders for Collaborative Filtering.CC BY-NC-ND 4.0 WWW 2018, April 23–27, 2018, Lyon, France

[Maciej Kula. (2015)](https://arxiv.org/pdf/1507.08439.pdf) Metadata Embeddings for User and Item Cold-start Recommendations. CBRecSys 2015, September 20, 2015, Vienna, Austria.

[Gupta, K., Raghuprasad, M. Y., & Kumar, P. (2018)](https://arxiv.org/pdf/1808.01006.pdf). A Hybrid Variational Autoencoder for Collaborative Filtering. *arXiv preprint arXiv:1808.01006*.