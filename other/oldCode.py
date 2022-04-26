# linkage_methods = ['ward', 'complete', 'average']
# affinity_methods = ['euclidean', 'manhattan', 'cosine']

def pipe_agglomerative_n_clusters(X_toFit):

    X_tmp = X_toFit.copy()
    for n_clusters in range(3, 10):
        for m in linkage_methods:
            for a in affinity_methods:
                skip = False
                if (m == 'ward') & (a != 'euclidean'):
                    continue
                clusterer = AgglomerativeClustering(n_clusters = n_clusters, affinity = a, linkage = m).fit(X_toFit)
                labels = clusterer.labels_
                score = round(silhouette_score(X_tmp, labels), 3)
                param_score = {'algorithm': m, 'n_clusters' : n_clusters,'distance' : 0, 'score' : score, 'affinity' : a,
                               'clusterer' : clusterer, 'df':n}
                X_tmp['cluster'] = labels

                for cluster in X_tmp['cluster'].unique():
                    portion = round(len(tmp[tmp.cluster == cluster])/len(tmp)*100, 2)
                    print(portion)
                    if portion < 2.00:
                        skip = True
                if skip:
                    continue
                print("skip", skip)
                list_scores.append(param_score)  


def pipe_agglomerative_distance(X_toFit):
    X_tmp = X_toFit.copy()
    for distance_threshold in [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        for m in linkage_methods:
            for a in affinity_methods:
                if (m == 'ward') & (a != 'euclidean'):
                    continue
                clusterer = AgglomerativeClustering(n_clusters = None, distance_threshold = distance_threshold, 
                                                    affinity = a, linkage = m).fit(X_toFit)
                labels = clusterer.labels_
                if len(np.unique(labels) < 3):
                    continue
                score = round(silhouette_score(X_tmp, labels), 3)
                print(score)
                param_score = {'algorithm': m, 'n_clusters' : np.unique(labels), 'distance' : distance_threshold, 
                               'score' : score, 'affinity' : a, 'clusterer' : clusterer, 'df':n}

                
                sns.set()
sns.set_color_codes()
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(50, 10))

for n, ax in enumerate(axes):
    
    if n == 0: 
        no_affinity = ['k-means', 'ward']
        palette = ['gold', 'tomato']
        data = clustering[clustering.algorithm.isin(no_affinity)].copy()
        ax.set_title('K-means (Partitional) vs Ward Linkage (Hierarchical agglomerative)', fontsize = 30, pad = 30, fontfamily = 'Times New Roman')
    if n == 1:
        palette = ['mediumseagreen', 'darkgreen', 'lightgreen']
        data = clustering[clustering.algorithm == 'complete'].copy()
        ax.set_title('Complete Linkage (Hierarchical agglomerative)', fontsize = 30,pad = 30,  fontfamily = 'Times New Roman')
    if n == 2:
        palette = ['dodgerblue', 'skyblue', 'navy']
        data = clustering[clustering.algorithm == 'average'].copy()
        ax.set_title('Average Linkage (Hierarchical agglomerative)', fontsize = 30, pad = 30, fontfamily = 'Times New Roman')
     
    if n > 0:
        sns.scatterplot(ax = ax, x='n_clusters', y="score", hue = 'affinity', palette = palette, s= 250, data=data)
    else:
        sns.scatterplot(ax = ax, x='n_clusters', y="score", hue = 'algorithm', palette = palette, s= 250, data = data)
    
    ax.set_ylim(0.35, 0.55)
    if n >0:
        ax.tick_params(axis='y', labelsize= 0, pad = 0)
        ax.set_ylabel('', fontsize = 25, labelpad = 30)
        ax.set_yticklabels([])
#         ax.axes.yaxis.set_visible(False)
    else:
        ax.set_ylabel('Silhouette score', fontsize = 25, labelpad = 30, fontfamily = 'Times New Roman')
        for tick in ax.get_yticklabels(): 
            tick.set_fontname('Times New Roman')
        
    for tick in ax.get_xticklabels(): tick.set_fontname('Times New Roman')
    ax.tick_params(axis='both', labelsize= 25, pad = 15)
    
    ax.legend(frameon=False, loc="upper left", markerscale = 2, labelspacing=1.5)
    leg = ax.get_legend()    
    plt.setp(leg.texts, family='Times New Roman', fontsize = 25)
    
    ax.set_xlabel('Nr clusters', fontsize = 25, labelpad = 30, fontfamily = 'Times New Roman')
    plt.rcParams['font.family'] = 'Times New Roman'
    
fig.subplots_adjust(wspace=0.1)


from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn.cluster import OPTICS, Birch, DBSCAN
eps_l = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
metrics = ['cityblock', 'cosine', 'euclidean', 'manhattan']

def pipe_dbscan(X_toFit):
    X_tmp = X_toFit.copy()
    for eps in eps_l:
        for sample in range(1,15):
            clusterer = DBSCAN(eps= eps, min_samples=sample, metric='euclidean', 
                              algorithm='auto').fit(X_toFit)
            labels = clusterer.labels_
            if check_counts(X_tmp, labels):
                continue
                
            score = round(silhouette_score(X_tmp, labels, metric='sqeuclidean'), 3)
            param_score = {'algorithm': 'dbscan', 'n_clusters':len(np.unique(labels)), 'score' : score, 'clusterer': clusterer}
            list_scores.append(param_score)
            
def pipe_optics(X_toFit):
    X_tmp = X_toFit.copy()
    for sample in range(1,15):
        for eps in eps_l:
            for metric in metrics:
                for algo in ['ball_tree', 'kd_tree', 'brute']:
                    if (metric == 'cosine'): continue                    
                    for c_m in ['xi', 'dbscan']:
                        clusterer = OPTICS(min_samples = sample, eps = eps, metric = metric, cluster_method = c_m, 
                                           algorithm=algo).fit(X_toFit)
                        labels = clusterer.labels_
                        if check_counts(X_tmp, labels):
                            continue

                        score = round(silhouette_score(X_tmp, labels, metric='sqeuclidean'), 3)
                        param_score = {'algorithm': 'optics', 'n_clusters' : len(np.unique(labels)), 'score' : score, 
                           'clusterer' : clusterer}
                        list_scores.append(param_score)
                    
def pipe_mixture(X_toFit):
    X_tmp = X_toFit.copy()
    for n_components in range(1,20):
        for cov in ['full', 'tied', 'diag', 'spherical']:
            clusterer = mixture.GaussianMixture(n_components= n_components, covariance_type=cov, init_params = 'random', 
                                                max_iter=1000).fit(X_toFit)
            labels = clusterer.predict(X_toFit)
            if check_counts(X_tmp, labels):
                continue

            score = round(silhouette_score(X_tmp, labels, metric='sqeuclidean'), 3)
            param_score = {'algorithm': 'gaussian', 'n_clusters' : len(np.unique(labels)),
                                                                       'score' : score, 'clusterer' : clusterer}
            list_scores.append(param_score)

def pipe_birch(X_toFit):
    X_tmp = X_toFit.copy()
    for n_clusters in range(3, 9):
        for threshold in eps_l:
            clusterer = Birch(threshold = threshold, n_clusters=n_clusters).fit(X_toFit)
            labels = clusterer.labels_
            if check_counts(X_tmp, labels):
                continue
                
            score = round(silhouette_score(X_tmp, labels, metric='sqeuclidean'), 3)
            param_score = {'algorithm': 'birch', 'n_clusters' : len(np.unique(labels)),
                                                                    'score' : score, 'clusterer' : clusterer}
            list_scores.append(param_score)
                