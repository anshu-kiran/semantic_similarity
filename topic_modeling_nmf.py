import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import sent_to_words, lemmatization

file_name = "data/quora_big_cleaned.csv"
df = pd.read_csv(file_name, sep=",")
df.dropna(inplace=True)

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=4000, stop_words='english')
vectorized_data = vectorizer.fit_transform(df['question_lemmatize_clean'].values.astype(str))

nmf = NMF(n_components=20, random_state=5, alpha=.1, l1_ratio=.5).fit(vectorized_data)
nmf_output = nmf.fit_transform(vectorized_data)

keywords = np.array(vectorizer.get_feature_names_out())
topic_keywords = []
for topic_weights in nmf.components_:
    top_keyword_locs = (-topic_weights).argsort()[:15]
    topic_keywords.append(keywords.take(top_keyword_locs))

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

text = [
    "Just like Larry Page and Sergey Brin unseated their incumbents with a better search engine, how likely is it "
    "that two Computer Science PhD students create a search engine that unseats Google? How vulnerable is Google to "
    "this possibility?"]

text_cleaned = list(sent_to_words(text))
text_cleaned = lemmatization(text_cleaned, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
text_cleaned = vectorizer.transform(text_cleaned)

topic_probability_scores = nmf.transform(text_cleaned)
topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()

print(topic, topic_probability_scores)
