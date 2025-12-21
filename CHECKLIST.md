# CHECKLIST

## linalg

- [x] `np.dot` → `tinyops.linalg.dot`
- [x] `np.matmul` → `tinyops.linalg.matmul`
- [x] `np.vdot` → `tinyops.linalg.vdot`
- [x] `np.inner` → `tinyops.linalg.inner`
- [x] `np.outer` → `tinyops.linalg.outer`
- [x] `np.tensordot` → `tinyops.linalg.tensordot`
- [x] `np.einsum` → `tinyops.linalg.einsum`
- [x] `np.linalg.inv` → `tinyops.linalg.inv`
- [x] `np.linalg.pinv` → `tinyops.linalg.pinv`
- [x] `np.linalg.solve` → `tinyops.linalg.solve`
- [x] `np.linalg.lstsq` → `tinyops.linalg.lstsq`
- [ ] `np.linalg.det` → `tinyops.linalg.det`
- [x] `np.linalg.norm` → `tinyops.linalg.norm`
- [x] `np.linalg.cond` → `tinyops.linalg.cond`
- [ ] `np.linalg.matrix_rank` → `tinyops.linalg.matrix_rank`
- [x] `np.trace` → `tinyops.linalg.trace`
- [ ] `np.linalg.eig` → `tinyops.linalg.eig`
- [ ] `np.linalg.eigh` → `tinyops.linalg.eigh`
- [ ] `np.linalg.eigvals` → `tinyops.linalg.eigvals`
- [ ] `np.linalg.svd` → `tinyops.linalg.svd`
- [ ] `np.linalg.qr` → `tinyops.linalg.qr`
- [ ] `np.linalg.cholesky` → `tinyops.linalg.cholesky`
- [x] `np.linalg.matrix_power` → `tinyops.linalg.matrix_power`
- [x] `np.kron` → `tinyops.linalg.kron`
- [x] `np.diagonal` → `tinyops.linalg.diagonal`

## stats

- [x] `np.mean` → `tinyops.stats.mean`
- [x] `np.median` → `tinyops.stats.median`
- [x] `np.std` → `tinyops.stats.std`
- [x] `np.var` → `tinyops.stats.var`
- [x] `np.average` → `tinyops.stats.average`
- [x] `np.percentile` → `tinyops.stats.percentile`
- [x] `np.quantile` → `tinyops.stats.quantile`
- [x] `np.ptp` → `tinyops.stats.ptp`
- [x] `np.histogram` → `tinyops.stats.hist`
- [ ] `np.histogram2d` → `tinyops.stats.hist2d`
- [ ] `np.histogramdd` → `tinyops.stats.histdd`
- [x] `np.bincount` → `tinyops.stats.bincount`
- [ ] `np.digitize` → `tinyops.stats.digitize`
- [ ] `np.corrcoef` → `tinyops.stats.corrcoef`
- [ ] `np.correlate` → `tinyops.stats.correlate`
- [ ] `np.cov` → `tinyops.stats.cov`

## image

- [ ] `cv2.resize` → `tinyops.image.resize`
- [ ] `cv2.rotate` → `tinyops.image.rotate`
- [ ] `cv2.flip` → `tinyops.image.flip`
- [ ] `cv2.warpAffine` → `tinyops.image.warp_affine`
- [ ] `cv2.warpPerspective` → `tinyops.image.warp_perspective`
- [x] `cv2.blur` → `tinyops.image.blur`
- [x] `cv2.boxFilter` → `tinyops.image.box_filter`
- [x] `cv2.GaussianBlur` → `tinyops.image.gaussian_blur`
- [ ] `cv2.medianBlur` → `tinyops.image.median_blur`
- [ ] `cv2.bilateralFilter` → `tinyops.image.bilateral_filter`
- [ ] `cv2.filter2D` → `tinyops.image.filter2d`
- [ ] `cv2.Sobel` → `tinyops.image.sobel`
- [ ] `cv2.Scharr` → `tinyops.image.scharr`
- [ ] `cv2.Laplacian` → `tinyops.image.laplacian`
- [ ] `cv2.Canny` → `tinyops.image.canny`
- [ ] `cv2.dilate` → `tinyops.image.dilate`
- [ ] `cv2.erode` → `tinyops.image.erode`
- [ ] `cv2.morphologyEx` → `tinyops.image.morphology`
- [ ] `cv2.threshold` → `tinyops.image.threshold`
- [ ] `cv2.adaptiveThreshold` → `tinyops.image.adaptive_threshold`
- [ ] `cv2.cvtColor` → `tinyops.image.cvt_color`
- [ ] `cv2.equalizeHist` → `tinyops.image.equalize_hist`
- [ ] `cv2.normalize` → `tinyops.image.normalize`
- [ ] `torchvision.transforms.CenterCrop` → `tinyops.image.center_crop`
- [ ] `torchvision.transforms.Pad` → `tinyops.image.pad`
- [ ] `torchvision.transforms.ColorJitter` → `tinyops.image.color_jitter`

## audio

- [ ] `torchaudio.transforms.Spectrogram` → `tinyops.audio.spectrogram`
- [ ] `torchaudio.transforms.MelSpectrogram` → `tinyops.audio.mel_spectrogram`
- [ ] `torchaudio.transforms.MFCC` → `tinyops.audio.mfcc`
- [ ] `torchaudio.transforms.MelScale` → `tinyops.audio.mel_scale`
- [ ] `torchaudio.transforms.AmplitudeToDB` → `tinyops.audio.amplitude_to_db`
- [ ] `torchaudio.transforms.GriffinLim` → `tinyops.audio.griffin_lim`
- [ ] `torchaudio.transforms.Resample` → `tinyops.audio.resample`
- [ ] `torchaudio.transforms.MuLawEncoding` → `tinyops.audio.mu_law_encode`
- [ ] `torchaudio.transforms.MuLawDecoding` → `tinyops.audio.mu_law_decode`
- [ ] `torchaudio.transforms.TimeStretch` → `tinyops.audio.time_stretch`
- [ ] `torchaudio.transforms.FrequencyMasking` → `tinyops.audio.freq_mask`
- [ ] `torchaudio.transforms.TimeMasking` → `tinyops.audio.time_mask`
- [ ] `torchaudio.transforms.Fade` → `tinyops.audio.fade`

## signal

- [ ] `np.fft.fft` → `tinyops.signal.fft`
- [ ] `np.fft.ifft` → `tinyops.signal.ifft`
- [ ] `np.fft.fft2` → `tinyops.signal.fft2`
- [ ] `np.fft.ifft2` → `tinyops.signal.ifft2`
- [ ] `np.fft.rfft` → `tinyops.signal.rfft`
- [ ] `np.fft.irfft` → `tinyops.signal.irfft`
- [ ] `np.fft.fftfreq` → `tinyops.signal.fftfreq`
- [ ] `np.convolve` → `tinyops.signal.convolve`
- [ ] `np.correlate` → `tinyops.signal.correlate`
- [ ] `np.hanning` → `tinyops.signal.hanning`
- [ ] `np.hamming` → `tinyops.signal.hamming`
- [ ] `np.blackman` → `tinyops.signal.blackman`
- [ ] `np.kaiser` → `tinyops.signal.kaiser`

## io

- [ ] wav decode → `tinyops.io.decode_wav`
- [ ] wav encode → `tinyops.io.encode_wav`
- [ ] bmp decode → `tinyops.io.decode_bmp`
- [ ] bmp encode → `tinyops.io.encode_bmp`

## ml

### preprocessing

- [ ] `sklearn.preprocessing.StandardScaler` → `tinyops.ml.standard_scaler`
- [ ] `sklearn.preprocessing.MinMaxScaler` → `tinyops.ml.minmax_scaler`
- [ ] `sklearn.preprocessing.MaxAbsScaler` → `tinyops.ml.maxabs_scaler`
- [ ] `sklearn.preprocessing.RobustScaler` → `tinyops.ml.robust_scaler`
- [ ] `sklearn.preprocessing.Normalizer` → `tinyops.ml.normalizer`
- [ ] `sklearn.preprocessing.Binarizer` → `tinyops.ml.binarizer`
- [ ] `sklearn.preprocessing.OneHotEncoder` → `tinyops.ml.onehot_encoder`
- [ ] `sklearn.preprocessing.LabelEncoder` → `tinyops.ml.label_encoder`
- [ ] `sklearn.preprocessing.PolynomialFeatures` → `tinyops.ml.polynomial_features`

### decomposition

- [ ] `sklearn.decomposition.PCA` → `tinyops.ml.pca`
- [ ] `sklearn.decomposition.TruncatedSVD` → `tinyops.ml.truncated_svd`
- [ ] `sklearn.decomposition.NMF` → `tinyops.ml.nmf`
- [ ] `sklearn.decomposition.FastICA` → `tinyops.ml.fastica`
- [ ] `sklearn.decomposition.FactorAnalysis` → `tinyops.ml.factor_analysis`

### cluster

- [ ] `sklearn.cluster.KMeans` → `tinyops.ml.kmeans`
- [ ] `sklearn.cluster.MiniBatchKMeans` → `tinyops.ml.minibatch_kmeans`
- [ ] `sklearn.cluster.DBSCAN` → `tinyops.ml.dbscan`
- [ ] `sklearn.cluster.AgglomerativeClustering` → `tinyops.ml.agglomerative`
- [ ] `sklearn.cluster.SpectralClustering` → `tinyops.ml.spectral_clustering`
- [ ] `sklearn.cluster.MeanShift` → `tinyops.ml.mean_shift`

### linear_model

- [ ] `sklearn.linear_model.LinearRegression` → `tinyops.ml.linear_regression`
- [ ] `sklearn.linear_model.Ridge` → `tinyops.ml.ridge`
- [ ] `sklearn.linear_model.Lasso` → `tinyops.ml.lasso`
- [ ] `sklearn.linear_model.ElasticNet` → `tinyops.ml.elastic_net`
- [ ] `sklearn.linear_model.LogisticRegression` → `tinyops.ml.logistic_regression`
- [ ] `sklearn.linear_model.SGDClassifier` → `tinyops.ml.sgd_classifier`
- [ ] `sklearn.linear_model.SGDRegressor` → `tinyops.ml.sgd_regressor`

### neighbors

- [ ] `sklearn.neighbors.KNeighborsClassifier` → `tinyops.ml.knn_classifier`
- [ ] `sklearn.neighbors.KNeighborsRegressor` → `tinyops.ml.knn_regressor`
- [ ] `sklearn.neighbors.NearestNeighbors` → `tinyops.ml.nearest_neighbors`

### naive_bayes

- [ ] `sklearn.naive_bayes.GaussianNB` → `tinyops.ml.gaussian_nb`
- [ ] `sklearn.naive_bayes.MultinomialNB` → `tinyops.ml.multinomial_nb`
- [ ] `sklearn.naive_bayes.BernoulliNB` → `tinyops.ml.bernoulli_nb`

### svm

- [ ] `sklearn.svm.SVC` → `tinyops.ml.svc`
- [ ] `sklearn.svm.SVR` → `tinyops.ml.svr`
- [ ] `sklearn.svm.LinearSVC` → `tinyops.ml.linear_svc`
- [ ] `sklearn.svm.LinearSVR` → `tinyops.ml.linear_svr`

### tree

- [ ] `sklearn.tree.DecisionTreeClassifier` → `tinyops.ml.decision_tree_classifier`
- [ ] `sklearn.tree.DecisionTreeRegressor` → `tinyops.ml.decision_tree_regressor`

### ensemble

- [ ] `sklearn.ensemble.RandomForestClassifier` → `tinyops.ml.random_forest_classifier`
- [ ] `sklearn.ensemble.RandomForestRegressor` → `tinyops.ml.random_forest_regressor`
- [ ] `sklearn.ensemble.GradientBoostingClassifier` → `tinyops.ml.gradient_boosting_classifier`
- [ ] `sklearn.ensemble.GradientBoostingRegressor` → `tinyops.ml.gradient_boosting_regressor`
- [ ] `sklearn.ensemble.AdaBoostClassifier` → `tinyops.ml.adaboost_classifier`
- [ ] `sklearn.ensemble.AdaBoostRegressor` → `tinyops.ml.adaboost_regressor`

### metrics

- [ ] `sklearn.metrics.accuracy_score` → `tinyops.ml.accuracy`
- [ ] `sklearn.metrics.precision_score` → `tinyops.ml.precision`
- [ ] `sklearn.metrics.recall_score` → `tinyops.ml.recall`
- [ ] `sklearn.metrics.f1_score` → `tinyops.ml.f1`
- [ ] `sklearn.metrics.confusion_matrix` → `tinyops.ml.confusion_matrix`
- [ ] `sklearn.metrics.roc_auc_score` → `tinyops.ml.roc_auc`
- [ ] `sklearn.metrics.mean_squared_error` → `tinyops.ml.mse`
- [ ] `sklearn.metrics.mean_absolute_error` → `tinyops.ml.mae`
- [ ] `sklearn.metrics.r2_score` → `tinyops.ml.r2`
- [ ] `sklearn.metrics.silhouette_score` → `tinyops.ml.silhouette`

### manifold

- [ ] `sklearn.manifold.TSNE` → `tinyops.ml.tsne`
- [ ] `sklearn.manifold.Isomap` → `tinyops.ml.isomap`
- [ ] `sklearn.manifold.MDS` → `tinyops.ml.mds`

### mixture

- [ ] `sklearn.mixture.GaussianMixture` → `tinyops.ml.gmm`
- [ ] `sklearn.mixture.BayesianGaussianMixture` → `tinyops.ml.bayesian_gmm`

## text

- [ ] `Levenshtein.distance` → `tinyops.text.levenshtein_distance`
- [ ] `nltk.edit_distance` → `tinyops.text.edit_distance`
- [ ] `sklearn.metrics.pairwise_distances(metric='hamming')` → `tinyops.text.pairwise_hamming_distance`
- [ ] `scipy.spatial.distance.hamming` → `tinyops.text.hamming_distance`
- [ ] `difflib.SequenceMatcher.ratio` → `tinyops.text.sequence_matcher_ratio`
- [ ] `rapidfuzz.distance.Levenshtein.distance` → `tinyops.text.levenshtein_distance`
- [ ] `jellyfish.levenshtein_distance` → `tinyops.text.levenshtein_distance`
- [ ] `jellyfish.jaro_winkler_similarity` → `tinyops.text.jaro_winkler_similarity`
- [ ] `sklearn.feature_extraction.text.TfidfVectorizer` → `tinyops.text.tfidf_vectorizer`
- [ ] `sklearn.feature_extraction.text.CountVectorizer` → `tinyops.text.count_vectorizer`
