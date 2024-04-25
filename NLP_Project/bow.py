from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from numpy.linalg import norm
import numpy as np
import string
import nltk

stop_words = set(stopwords.words("english"))
class BagOfWord:
    def fit_transform(self, text1: str, text2: str):
        text1 = text1.lower()
        text2 = text2.lower()
        t1 = nltk.word_tokenize(text1)
        t2 = nltk.word_tokenize(text2)
        mp, mpA, mpB = {}, {}, {}
        for val in t1:
            if val not in string.punctuation and val not in stop_words:
                mp[val] = 0
                if val not in mpA: mpA[val] = 1
                else: mpA[val] += 1
        for val in t2:
            if val not in string.punctuation and val not in stop_words:
                mp[val] = 0
                if val in mpB: mpB[val] += 1
                else: mpB[val] = 1
        vectorA, vectorB = [], []
        for key in mp.keys():
            if key in mpA: vectorA.append(mpA[key])
            else: vectorA.append(0)
            if key in mpB: vectorB.append(mpB[key])
            else: vectorB.append(0)
        return np.array(vectorA), np.array(vectorB)


def cosineSimilarity(vectorA, vectorB) -> float:
    return np.dot(vectorA, vectorB) / (norm(vectorA) * norm(vectorB))

def bag_of_word_cosine():
    text1 = "single document"
    text2 = "single document"
    print("Code Implement")
    vectorize = BagOfWord()
    X = vectorize.fit_transform(text1, text2)
    print("Vector of text 1: {}".format(X[0]))
    print("Vector of text 2: {}".format(X[1]))
    print("Cosine similarity: {}".format(cosineSimilarity(X[0], X[1])))

    print('\n')
    print("Code sklearn")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text1, text2])
    print("Vector of text 1: {}".format(X[0].toarray()))
    print("Vector of text 2: {}".format(X[1].toarray()))
    cosine = cosine_similarity(X[0], X[1])
    print("Cosine similarity: {}".format(cosine[0][0]))


def main():
    document = """
    Abstract: Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). 
    The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. 
    By combining the insights from our exploration with scale and our new ``Colossal Clean Crawled Corpus'', we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. 
    To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.Recently, it has become increasingly common to pre-train the entire model on a data-rich
    task. Ideally, this pre-training causes the model to develop general-purpose abilities and
    knowledge that can then be transferred to downstream tasks. In applications of transfer
    learning to computer vision (Oquab et al., 2014; Jia et al., 2014; Huh et al., 2016; Yosinski
    et al., 2014), pre-training is typically done via supervised learning on a large labeled data set
    like ImageNet (Russakovsky et al., 2015; Deng et al., 2009). In contrast, modern techniques
    for transfer learning in NLP often pre-train using unsupervised learning on unlabeled data.
    This approach has recently been used to obtain state-of-the-art results in many of the most
    common NLP benchmarks (Devlin et al., 2018; Yang et al., 2019; Dong et al., 2019; Liu
    et al., 2019c; Lan et al., 2019). Beyond its empirical strength, unsupervised pre-training
    for NLP is particularly attractive because unlabeled text data is available en masse thanks
    to the Internet—for example, the Common Crawl project2 produces about 20TB of text
    data extracted from web pages each month. This is a natural fit for neural networks, which
    have been shown to exhibit remarkable scalability, i.e. it is often possible to achieve better
    performance simply by training a larger model on a larger data set (Hestness et al., 2017;
    Shazeer et al., 2017; Jozefowicz et al., 2016; Mahajan et al., 2018; Radford et al., 2019;
    Shazeer et al., 2018; Huang et al., 2018b; Keskar et al., 2019a)
    """.strip()
    key = "encounter cases"
    key_size = len(nltk.word_tokenize(key))
    bow = BagOfWord()
    vectorizer = CountVectorizer()
    spl = document.split()
    for idx in range(len(spl) - key_size):
        sub_text = ' '.join(t for t in spl[idx: idx + key_size]).strip()
        vector = vectorizer.fit_transform([sub_text, key])
        if np.sum(vector[0]) == 0: continue
        cosine = cosine_similarity(vector[0], vector[1])
        if cosine[0][0] == 0.9999999999999998 or cosine[0][0] == 1.0: print(f"Code sklearn: {sub_text}")
        vectorA, vectorB = bow.fit_transform(sub_text, key)
        if np.sum(vectorA) == 0: continue
        cosine = cosineSimilarity(vectorA, vectorB)
        if cosine == 0.9999999999999998 or cosine == 1.0: print(f"Code implement: {sub_text}")

if __name__ == "__main__":
    main()