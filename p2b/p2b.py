import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

folderPath = 'Labels'

files = [file for file in os.listdir(folderPath) if file.endswith(".xlsx")]

combinedFile = pd.DataFrame()

kappaScores = {}
fileKappaScores = {}
comparisonKappaScores = {}
isMerged = None

chunk_size = 1000  # Adjust the chunk size based on available memory

for file in files:
    filePath = os.path.join(folderPath, file)

    # Determine the row where the actual data starts
    skip_rows = 0
    while pd.read_excel(filePath, engine='openpyxl', skiprows=skip_rows, nrows=1).empty:
        skip_rows += 1

    # Read data in chunks
    chunks = pd.read_excel(filePath, engine='openpyxl', skiprows=skip_rows, nrows=chunk_size)
    for chunk in np.array_split(chunks, len(chunks) // chunk_size):
        # Process the chunk here
        for index, row in chunk.iterrows():
            if not pd.isnull(row['First violation']):
                if pd.isnull(row['Second violation']):
                    chunk.at[index, 'Second violation'] = 'No violation'
                if pd.isnull(row['Third violation']):
                    chunk.at[index, 'Third violation'] = 'No violation'

        fileName = os.path.splitext(os.path.basename(filePath))[0]

        columnNameOne = fileName + " First Violation"
        columnNameTwo = fileName + " Second Violation"
        columnNameThree = fileName + " Third Violation"

        chunk = chunk.rename(columns={'First violation': columnNameOne,
                                      'Second violation': columnNameTwo,
                                      'Third violation': columnNameThree})

        if isMerged is None:
            isMerged = chunk
        else:
            isMerged = pd.merge(isMerged, chunk, on=['title', 'review'], how='inner')

        # Update skip_rows for the next iteration
        skip_rows += chunk_size

is_merged = is_merged.drop_duplicates(subset=['title', 'review'], keep='first')
print('Merged File Made!')

# File with merged files
is_merged.to_excel("isMerged.xlsx", index=False, engine='openpyxl')

for review in is_merged['review'].unique():
    review_data = is_merged[is_merged['review'] == review]
    review_data = review_data.copy()

    category_mapping = {'No violation': 0, 'Visibility': 1, 'Free Will': 2, 'Truth': 3, 'Capacity': 4, 'Cognition': 5,
                        'Attention': 6, 'Statues': 7, 'Power': 8, 'Honesty': 9}

    columns = review_data.columns[2:]
    
    for i in range(0, len(columns)-2, 3):
        for j in range(i + 3, len(columns), 3):
            set_one = review_data[columns[i:i+3]].values.flatten()
            set_two = review_data[columns[j:j+3]].values.flatten()
            
            calculate_kappa_scores(file_kappa_scores, comparison_kappa_scores, set_one, set_two, columns[i].split()[0])

print("file_kappa_scores:", file_kappa_scores)
print("comparison_kappa_scores:", comparison_kappa_scores)

average_kappa_score = {file_name: total_kappa / count if not pd.isna(total_kappa) and count != 0 else float('nan')
                       for file_name, total_kappa, count in zip(file_kappa_scores.keys(),
                                                               file_kappa_scores.values(),
                                                               comparison_kappa_scores.values())}

print("average_kappa_score:", average_kappa_score)

sorted_files = sorted(average_kappa_score, key=average_kappa_score.get)

# Top 10 lowest
top_ten_lowest = sorted_files[:10]

# Top 5 lowest
top_five_lowest = sorted_files[:5]
print("Top 5 Files With The Lowest Kappa Scores:", ', '.join(top_five_lowest))

# Remove top 5 fakest files
is_merged = is_merged[~is_merged['review'].isin(top_five_lowest)]

for review in is_merged['review'].unique():
    review_data = is_merged[is_merged['review'] == review]
    majority_values = review_data.iloc[:, 2:].iloc[0]

    top_three_categories = majority_values.value_counts().nlargest(3)

    for col in review_data.columns[2:]:
        is_merged.loc[is_merged['review'] == review, col] = majority_values[col]

is_merged = is_merged.iloc[:, :5]

is_merged.rename(columns={is_merged.columns[2]: 'First violation',
                          is_merged.columns[3]: 'Second violation',
                          is_merged.columns[4]: 'Third violation'}, inplace=True)

is_merged.to_excel("isMerged_With_Top_Three_Categories.xlsx", index=False, engine='openpyxl')

all_reviews = ' '.join(is_merged['review'])

# NLTK to get sentiment scores
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(all_reviews)

# tokenize and remove stop words
stop_words = set(stopwords.words('english'))
tokenized_words = [word.lower() for word in word_tokenize(all_reviews) if word.isalpha() and word.lower() not in stop_words]

# Count how many times each word is used
word_count = Counter(tokenized_words)

# Sort from largest to smallest
sorted_counts = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# Display top words
top_words = sorted_counts[:20]
for word, count in top_words:
    print(f'{word}: {count} times')

category_counts = is_merged.iloc[:, 2:5].stack().value_counts()
top_three_categories = category_counts.head(3).index
print("Top Three Categories", top_three_categories)

for category in top_three_categories:
    tokenized_words = [word.lower() for word in word_tokenize(' '.join(is_merged[is_merged['review'].str.contains(category)]['review']))
                       if word.isalpha() and word.lower() not in stop_words]

    word_count = Counter(tokenized_words)

    sorted_counts = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    top_words = sorted_counts[:40]

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {category}")
    plt.axis('off')
    plt.show()
