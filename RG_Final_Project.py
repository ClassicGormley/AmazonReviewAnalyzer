import csv
import nltk
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob


subprocess.call('scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews.py -o reviews.'
                'csv', shell=True)


def get_word_freq(sentences):
    freqs = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word in punct:
                break
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1
    return freqs


def plot_zipf(my_dict, title):
    y_val = [x[1] for x in my_dict]

    #plt.yscale("log")
    #plt.xscale("log")
    plt.title("Zipf Curve " + title)
    plt.ylabel("Frequency")
    plt.xlabel("Word index")
    plt.plot(y_val)

    plt.show()


punct = [',', '.', "’", '"', '?', '@', ':', '#', '...', '!', 'https']

# Convert .csv file to .txt and save
with open('reviews.txt', "w") as output_file:
    with open('reviews.csv', "r") as my_input_file:
        [output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    output_file.close()


# Save reviews in a list removing the empty spaces in the .txt file
review_list = []
sentence_list = []
with open('reviews.txt', 'r', encoding='utf8') as read_file:
    review_list = read_file.readlines()
    review_list.pop(0)
    for i, review in enumerate(review_list):
        if review == '            \n':
            review_list.pop(i)


# Test the word frequency of the sentences in each review and plot them on a Zipf graph. Remove reviews that with poor
# word frequency as unhelpful.
for review in review_list:
    i = 0
    sentence_list = nltk.sent_tokenize(review)
    reuters_freq = get_word_freq(sentence_list)
    reuters_dict = sorted(reuters_freq.items(), key=lambda x: x[1], reverse=True)
    if int(reuters_dict[0][1]) < 5:
        review_list.pop(i)
        continue
    #print(reuters_dict)
    plot_zipf(reuters_dict, 'Review ' + str(i+1))
    i += 1

# Bigrams most frequent

for review in review_list:
    sentence_list = nltk.sent_tokenize(review)
    for sentence in sentence_list:
        bigram_fd = nltk.FreqDist(nltk.bigrams(sentence))

        print(bigram_fd.most_common())


# POS Tagging
review_pos_tags = {}
i = 0
for review in review_list:
    review_num = "Review " + str(i + 1)
    sentence_list = nltk.sent_tokenize(review)
    count_pos = Counter()
    for sentence in sentence_list:
        token = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(token)
        count_pos = Counter(tag for word, tag in pos_tags)
    review_pos_tags[review_num] = count_pos
    i += 1
print("POS Tags")
print(review_pos_tags)

# NER Tagging
review_ner_tags = {}
for review in review_list:
    review_num = "Review " + str(i + 1)
    sentence_list = nltk.sent_tokenize(review)
    token = []
    tagged = []
    ne_tags = []
    for sentence in sentence_list:
        token.append(nltk.word_tokenize(sentence))
        for token in token:
            tagged.append(nltk.pos_tag(token))
        for tag in tagged:
            ne_tags = nltk.ne_chunk(tag)
    review_ner_tags[review_num] = ne_tags
    i += 1
print('NER Tags')
print(review_ner_tags)

# Sentiment Analysis
i = 0
review_sentiment = {}
for review in review_list:
    tb = TextBlob(review)
    review_sentiment['Review ' + str(i+1)] = tb.sentiment.polarity
    i += 1

sorted_review_sentiment = sorted(review_sentiment.items(), key=lambda x: x[1])
print('Sentiment')
print(sorted_review_sentiment)
