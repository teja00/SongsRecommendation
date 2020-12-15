# Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")

from gensim.models import Word2Vec 
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import stats


import math
import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle 
FOLDER_PATH = "./dataset/yes_complete/"

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss


songs = pd.read_csv(FOLDER_PATH+"song_hash.txt", sep = '\t', header = None,
                    names = ['song_id', 'title', 'artist'], index_col = 0)
songs['artist - title'] = songs['artist'] + " - " + songs['title']
songs
def readTXT(filename, start_line=0, sep=None):
    with open(FOLDER_PATH+filename) as file:
        return [line.rstrip().split(sep) for line in file.readlines()[start_line:]]
tags = readTXT("tags.txt")
tags[7:12]
mapping_tags = dict(readTXT("tag_hash.txt", sep = ', '))
mapping_tags['#'] = "unknown"

song_tags = pd.DataFrame({'tag_names': [list(map(lambda x: mapping_tags.get(x), t)) for t in tags]})
song_tags.index.name = 'song_id'
songs = pd.merge(left = songs, right = song_tags, how = 'left',
                 left_index = True, right_index = True)
songs.index = songs.index.astype('str')
songs.head()

unknown_songs = songs[(songs['artist'] == '-') | (songs['title'] == '-')]
songs.drop(unknown_songs.index, inplace = True)

playlist = readTXT("train.txt", start_line = 2) + readTXT("test.txt", start_line = 2)
print(f"Playlist Count: {len(playlist)}")

for i in range(0, 3):
    print("-------------------------")
    print(f"Playlist Idx. {i}: {len(playlist[i])} Songs")
    print("-------------------------")
    print(playlist[i])

playlist_wo_unknown = [[song_id for song_id in p if song_id not in unknown_songs.index]
                       for p in playlist]


clean_playlist = [p for p in playlist_wo_unknown if len(p) > 1]
print(f"Playlist Count After Cleansing: {len(clean_playlist)}")

#unique_songs = set(itertools.chain.from_iterable(clean_playlist))
#song_id_not_exist = set(songs.index) - unique_songs
#songs.drop(song_id_not_exist, inplace = True)
songs = songs.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print(f"Unique Songs After Cleansing: {songs.shape[0]}")

MODEL_PATH = "model/"
playlist_train, playlist_test = train_test_split(clean_playlist, test_size = 60,
                                                 shuffle = True, random_state = 123)


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

model = Word2Vec(
    size = 256,
    window = 10,
    min_count = 1,
    sg = 0,
    negative = 20,
    workers = multiprocessing.cpu_count()-1)
print(model)

logging.disable(logging.NOTSET) # enable logging
t = time()

model.build_vocab(playlist_train)

print(f"Time to build vocab: {round((time() - t), 2)} seconds")

logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

model.train(playlist_train,
            total_examples = model.corpus_count,
            epochs = 100,
            compute_loss = True,
            callbacks = [callback]) 

print(f"Time to train the model: {round((time() - t), 2)} seconds")

print(model)
model.save(MODEL_PATH+"song2vec.model")

logging.disable(logging.INFO) # disable logging
model = Word2Vec.load(MODEL_PATH+"song2vec.model")

plt.plot(range(1, model.epochs+1), model.callbacks[0].training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss", fontweight = "bold")
plt.show()


fig, axes = plt.subplots(6, 1, figsize = (50, 30))

slug = '4162'
song_id_list = [(slug, "Main Song"), *[t for t in model.wv.most_similar(slug)[:5]]] 

for ax, (song_id, sim) in zip(axes.flat, song_id_list):
    ax.imshow([model.wv[song_id]], cmap = "binary", aspect = "auto")
    ax.set_title(songs.loc[song_id+".0", "artist - title"], fontsize = 50)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"Similarity:\n{sim:.3f}" if sim != song_id_list[0][1] else sim,
                  rotation = "horizontal", ha = "left", va = "center", fontsize = 50)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

def meanVectors(playlist):
    vec = []
    for song_id in playlist:
        try:
            vec.append(model.wv[song_id])
        except:
            continue
    return np.mean(vec, axis=0)

playlist_vec = list(map(meanVectors, playlist_test))

def similarSongsByVector(vec, n = 10, by_name = True):
    # extract most similar songs for the input vector
    similar_songs = model.wv.similar_by_vector(vec, topn = n)
    
    # extract name and similarity score of the similar products
    if by_name:
        similar_songs = [(songs.loc[song_id+".0", "artist - title"], sim)
                              for song_id, sim in similar_songs]
    
    return similar_songs

def print_recommended_songs(idx, n):
    print("============================")
    print("SONGS PLAYLIST")
    print("============================")
    for song_id in playlist_test[idx]:
        print(songs.loc[song_id+'.0', "artist - title"])
    print()
    print("============================")
    print(f"TOP {n} RECOMMENDED SONGS")
    print("============================")
    for song, sim in similarSongsByVector(playlist_vec[idx], n):
        print(f"[Similarity: {sim:.3f}] {song}")
    print("============================")
print_recommended_songs(idx = 10 , n = 10)

top_n_songs = 25

def hitRateRandom(playlist, n_songs):
    hit = 0
    for i, target in enumerate(playlist):
        random.seed(i)
        recommended_songs = random.sample(list(songs.index), n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)

eval_random = pd.Series([hitRateRandom(p, n_songs = top_n_songs)
                         for p in tqdm(playlist_test, position=0, leave=True)])
eval_random.mean()

mapping_tag2song = songs.explode('tag_names').reset_index().groupby('tag_names')['song_id'].apply(list)
mapping_tag2song 

def hitRateContextSongTag(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w]+'.0' for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for i, (context, target) in enumerate(context_target_list):
        try:   
            context_song_tags = set(songs.loc[context, 'tag_names'].explode().values)
            possible_songs_id = set(mapping_tag2song[context_song_tags].explode().values)
        
            random.seed(i)
            recommended_songs = random.sample(possible_songs_id, n_songs)
            hit += int(target in recommended_songs)
        except:
            continue
    return hit/len(playlist)

eval_song_tag = pd.Series([hitRateContextSongTag(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song_tag.mean()
def hitRateSong2Vec(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]

    for context, target in context_target_list:
        context_vector = meanVectors(context)
        recommended_songs = similarSongsByVector(context_vector, n = n_songs, by_name = False)
        songs_id = list(zip(*recommended_songs))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)

eval_song2vec = pd.Series([hitRateSong2Vec(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song2vec.mean()

eval_df = pd.concat([eval_random.rename("Random"),
           eval_song_tag.rename("Song Tag"),
           #eval_clust.rename("Clustering"),
           eval_song2vec.rename("Song2Vec")], axis = 1)
#save2Pickle(eval_df, "eval_df")

#eval_df = loadPickle("eval_df")
g = eval_df.mean().sort_values().plot(kind = 'barh')
g.set_xlabel("Average Hit Rate")
g.set_title("Recommender Evaluation", fontweight = "bold")
plt.show()



"""
np.random.seed(123)
random_vec = np.random.uniform(0, 1, (2500, 2))
skm_test = KMeans(n_clusters = 8, n_jobs = -1,
                           random_state = 123).fit(random_vec)

normalized_random_vec = random_vec/np.linalg.norm(random_vec, axis=1, keepdims=True)
cluster_df = pd.DataFrame({'x': random_vec[:,0],
                           'y': random_vec[:,1],
                           'x_proj': normalized_random_vec[:,0],
                           'y_proj': normalized_random_vec[:,1],
                           'cluster': skm_test.labels_})

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(10,10))

plot_list = [('x', 'y'), ('x_proj', 'y_proj'), ('x_proj', 'y_proj'), ('x', 'y')]
hue_list = [None, None, 'cluster', 'cluster']
title_list = ["1. RAW VECTOR DATA", "2. PROJECTED VECTOR ONTO A UNIT CIRCLE",
              "3. CLUSTERED PROJECTED VECTOR", "4. CLUSTERED VECTOR DATA"]

for ax, (x,y), hue, title in zip(axes.flat, plot_list, hue_list, title_list):
    sns.scatterplot(data = cluster_df,
                    x = x, y = y, hue = hue,
                    palette = "Reds", legend = False,
                    ax = ax)
    ax.set_title(title, fontweight = "bold")

plt.suptitle("ILLUSTRATION OF SPHERICAL K-MEANS CLUSTERING",
             fontweight = "bold", fontsize = 20, y = 1.02)
plt.tight_layout()
plt.show()


embedding_matrix = model.wv[model.wv.vocab.keys()]
embedding_matrix.shape


range_k_clusters = (10, 500)
skm_list = []
for k in tqdm(range(*range_k_clusters, 10)):
    skm = KMeans(n_clusters = k,
                          n_init = 5, n_jobs = -1,
                          random_state = 123).fit(embedding_matrix)
    
    result_dict = {
        "k": k,
        "WCSS": skm.inertia_,
        "skm_object": skm
    }
    
    skm_list.append(result_dict)
skm_df = pd.DataFrame(skm_list).set_index('k')
skm_df.head()
"""
def save2Pickle(obj, filename):
    with open(f"{MODEL_PATH}{filename}.pkl", "wb") as file:
        pickle.dump(obj, file)

def loadPickle(filename):
    with open(f"{MODEL_PATH}{filename}.pkl", "rb") as file:
        return pickle.load(file)
"""
save2Pickle(skm_df, "skm_cluster")

skm_df = loadPickle("skm_cluster")

skm_df.WCSS.plot()
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method", fontweight = "bold")
plt.show()

def locateOptimalElbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])
    
    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    
    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
    return x[np.argmax(dist)]

k_opt = locateOptimalElbow(skm_df.index, skm_df['WCSS'].values)
skm_opt = skm_df.loc[k_opt, "skm_object"]
skm_opt

songs_cluster = songs.copy()
model_keys = model.wv.vocab.keys()
model_keys_list = list(model_keys)
a = []
for x in model_keys_list:
    a.append(x+'.0')
count = 0
for i in a:
    try:
        songs_cluster.loc[i,'cluster'] = skm_opt.labels_[count]
        count += 1
    except:
        continue
#songs_cluster.loc[model_keys_list, 'cluster'] = skm_opt.labels_
songs_cluster['cluster'] = songs_cluster['cluster'].fillna(-1).astype('int').astype('category')

embedding_tsne = TSNE(n_components = 2, metric = 'cosine',
                      random_state = 123).fit_transform(embedding_matrix)

save2Pickle(embedding_tsne, "tsne_viz")

embedding_tsne = loadPickle("tsne_viz")
model_keys = model.wv.vocab.keys()
model_keys_list = list(model_keys)
a = []
for x in model_keys_list:
    a.append(x+'.0')
count = 0
for i in a:
    try:
        songs_cluster.loc[i, 'x'] = embedding_tsne[:,0][count]
        songs_cluster.loc[i, 'y'] = embedding_tsne[:,1][count]
        count += 1
    except:
        continue
        
sns.scatterplot(data = songs_cluster[songs_cluster['cluster'] != -1],
                x = 'x', y = 'y', palette = "viridis",
                hue = 'cluster', legend = False).set_title(f"{k_opt} Clusters of Song2Vec",
                                                           fontweight = "bold")
plt.show()


random.seed(100)
random_cluster2plot = random.sample(range(k_opt), 10)
random_songs = songs_cluster[songs_cluster.cluster.isin(random_cluster2plot)].copy()
random_songs_index = []
for i in random_songs.index:
    random_songs_index.append(i.strip(".0"))

random_tsne = TSNE(n_components = 2, metric = 'cosine',
                   random_state = 100).fit_transform(model.wv[random_songs_index])
random_songs.loc[random_songs.index, 'x'] = random_tsne[:,0]
random_songs.loc[random_songs.index, 'y'] = random_tsne[:,1]

g = sns.scatterplot(data = random_songs,
                x = 'x', y = 'y', palette = "viridis",
                hue = 'cluster')
g.legend(loc = "upper left", bbox_to_anchor = (1, 1))
g.set_title(f"Randomly selected {len(random_cluster2plot)} clusters of Song2Vec", fontweight = "bold")
plt.show()
"""
top_n_songs = 25

def hitRateRandom(playlist, n_songs):
    hit = 0
    for i, target in enumerate(playlist):
        random.seed(i)
        recommended_songs = random.sample(list(songs.index), n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)

eval_random = pd.Series([hitRateRandom(p, n_songs = top_n_songs)
                         for p in tqdm(playlist_test, position=0, leave=True)])
eval_random.mean()

mapping_tag2song = songs.explode('tag_names').reset_index().groupby('tag_names')['song_id'].apply(list)
mapping_tag2song 

def hitRateContextSongTag(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w]+'.0' for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for i, (context, target) in enumerate(context_target_list):
        try:   
            context_song_tags = set(songs.loc[context, 'tag_names'].explode().values)
            possible_songs_id = set(mapping_tag2song[context_song_tags].explode().values)
        
            random.seed(i)
            recommended_songs = random.sample(possible_songs_id, n_songs)
            hit += int(target in recommended_songs)
        except:
            continue
    return hit/len(playlist)

eval_song_tag = pd.Series([hitRateContextSongTag(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song_tag.mean()
"""
def hitRateClustering(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for context, target in context_target_list:
        cluster_numbers = skm_opt.predict([model.wv[c] for c in context if c in model.wv.vocab.keys()])
        majority_voting = stats.mode(cluster_numbers).mode[0]
        possible_songs_id = list(songs_cluster[songs_cluster['cluster'] == majority_voting].index)
        recommended_songs = random.sample(possible_songs_id, n_songs)
        songs_id = list(zip(*recommended_songs))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)

eval_clust = pd.Series([hitRateClustering(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_clust.mean()
"""
def hitRateSong2Vec(playlist, window, n_songs):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]

    for context, target in context_target_list:
        context_vector = meanVectors(context)
        recommended_songs = similarSongsByVector(context_vector, n = n_songs, by_name = False)
        songs_id = list(zip(*recommended_songs))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)

eval_song2vec = pd.Series([hitRateSong2Vec(p, model.window, n_songs = top_n_songs)
                           for p in tqdm(playlist_test, position=0, leave=True)])
eval_song2vec.mean()

eval_df = pd.concat([eval_random.rename("Random"),
           eval_song_tag.rename("Song Tag"),
           #eval_clust.rename("Clustering"),
           eval_song2vec.rename("Song2Vec")], axis = 1)
save2Pickle(eval_df, "eval_df")

eval_df = loadPickle("eval_df")
g = eval_df.mean().sort_values().plot(kind = 'barh')
g.set_xlabel("Average Hit Rate")
g.set_title("Recommender Evaluation", fontweight = "bold")
plt.show()

