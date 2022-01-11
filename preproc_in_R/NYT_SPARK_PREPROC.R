
setwd("./")

###########################################################
library(tm)

# deleteFunc function
deleteFunc <- content_transformer(function(x, pattern) {return (gsub(pattern, "", x))})

##########################################################################
# read and prepare data

df1 <-read.csv("../data/nyt_api_corpus.csv", head=TRUE, sep=",", encoding="utf-8", stringsAsFactors = FALSE)
df2 <-read.csv("../data/text_type.csv", head=TRUE, sep=";", encoding="utf-8", stringsAsFactors = FALSE)
df3 <-read.csv("../data/valid_start.csv", head=TRUE, sep=",", encoding="utf-8", stringsAsFactors = FALSE)

df3 <-df3[c("id", "majortopic")]

NYT_csv <- merge(df1, df2, by = "id")
NYT_csv <- merge(NYT_csv, df3, by = "id")

# set NA values to empty string
NYT_csv[is.na(NYT_csv)] <- ""

# remove caption
NYT_csv <- NYT_csv[NYT_csv$text_type == 'article',]

# remove missing title
NYT_csv <- NYT_csv[NYT_csv$title != '',]

# remove missing url
NYT_csv <- NYT_csv[NYT_csv$url_api != '',]

# create text column from title and original text column 
colnames(NYT_csv)[colnames(NYT_csv)=="id"] <- "doc_id"
data_all_2 <- within(NYT_csv,  text <- paste(title, lead_paragraph, headline_print, sep=" "))

# drop columns
data_lemma <- data_all_2[c("doc_id", "text")]

# create corpus
textcorp <- Corpus(DataframeSource(data_lemma))

# create processed text
textcorp <- tm_map(textcorp, removeNumbers)
textcorp <- tm_map(textcorp, removePunctuation)
textcorp <- tm_map(textcorp, stripWhitespace)
textcorp <- tm_map(textcorp, content_transformer(tolower))
textcorp <- tm_map(textcorp, removeWords, stopwords("english")) 
textcorp <- tm_map(textcorp, deleteFunc,"-")
textcorp <- tm_map(textcorp, deleteFunc, "–")
textcorp <- tm_map(textcorp, deleteFunc, "–")
textcorp <- tm_map(textcorp, deleteFunc, "—")
textcorp <- tm_map(textcorp, deleteFunc, "'")
textcorp <- tm_map(textcorp, deleteFunc, "ʻ")
textcorp <- tm_map(textcorp, deleteFunc, "ʼ")
textcorp <- tm_map(textcorp, deleteFunc, "ʽ")
textcorp <- tm_map(textcorp, deleteFunc, "„")
textcorp <- tm_map(textcorp, deleteFunc, "”")
textcorp <- tm_map(textcorp, deleteFunc, "\"")
textcorp <- tm_map(textcorp, deleteFunc, "\"")
textcorp <- tm_map(textcorp, stemDocument, language = "english")
textcorp <- tm_map(textcorp, removeWords, stopwords("english")) # once more because of stemming
textcorp <- tm_map(textcorp, stripWhitespace)

# create dataframe from corpus
data_lemma <- data.frame(text = sapply(textcorp, paste, collapse = " "), stringsAsFactors = FALSE)

# add id in new column
data_lemma["doc_id"]<-row.names(data_lemma)

# reattach processed text to original dataframe
data_all_2 <- data_all_2[c('doc_id', 'majortopic', 'date')]
data_all_with_text <- merge(data_all_2, data_lemma, by="doc_id")

# add column for simulation starting set separation
data_all_with_text$sim <- sample(3, size = nrow(data_all_with_text), replace = TRUE)

# write data to csv
write.table(data_all_with_text, file="../data/Data_NYT_clean_SPARK_START_sim.csv", sep=";", row.names = FALSE)
