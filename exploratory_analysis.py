import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

file_name = "data/quora_big_cleaned.csv"
df = pd.read_csv(file_name, sep=",")

plt.figure(1, figsize=(10, 6))
doc_lens = [len(d) for d in df.question_text]
plt.hist(doc_lens, bins=100)
plt.title('Distribution of Question character length')
plt.ylabel('Number of questions')
plt.xlabel('Question character length')

mpl.rcParams['figure.figsize'] = (12.0, 12.0)
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = .1
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=500,
                      max_font_size=40, random_state=42).generate(str(df['question_lemmatize_clean']))
print(wordcloud)
fig = plt.figure(2)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
