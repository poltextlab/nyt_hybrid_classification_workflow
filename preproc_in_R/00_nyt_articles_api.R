library(jsonlite)
library(dplyr)
library(stringr)
library(lubridate)
library(stopwords)
library(readr)


na_in_df <- function(df) {
    apply(df, 2, function(x) sum(is.na(x)))   
}

set.seed(042)


list_of_articles <- read.csv("../data/valid_start.csv", head = TRUE, sep = ",", encoding = "utf-8", stringsAsFactors = FALSE)

# helper function
remove_words <- function(str, stopwords) {
    x <- unlist(strsplit(str, " "))
    paste(x[!x %in% stopwords], collapse = " ")
}

eng_stop <- stopwords(language = "en")



# prepare date and title for api calls
list_of_articles <- list_of_articles %>% 
    ungroup() %>% 
    mutate(date = mdy(date), 
           date_q = str_remove_all(as.character(date), "-"),
           title_q = remove_words(title, eng_stop),
           title_q = str_remove_all(title, pattern = "[[:punct:]]"),
           title_q = str_to_lower(title_q),
           url_api = NA,
           abstract = NA,
           lead_paragraph = NA,
           print_section = NA,
           print_page = NA,
           headline_main = NA,
           headline_print = NA,
           keywords = NA,
           pub_date = NA,
           document_type = NA,
           news_desk = NA,
           section_name = NA,
           byline = NA,
           word_count = NA,
           api_hits = NA)


for (i in 1:nrow(list_of_articles)) {
    list_of_articles$title_q[i] <- remove_words(list_of_articles$title_q[i], eng_stop)
}

# reminder: NEVER PUBLISH YOUR CODE WITH THE PLAIN TEXT KEY!
nyt_key <- 'your new york times api key'

# API call limit
# https://developer.nytimes.com/faq#a11
# Yes, there are two rate limits per API: 4,000 requests per day and 10 requests per minute. You should sleep 6 seconds between calls to avoid hitting the per minute rate limit.

# function to access the NYTimes API
# version 2, with the implemented fixes
nyt_api_query <- function(df, range, verbose = FALSE) {
    for (i in range) {
        tryCatch({
            
            query <- str_c(unlist(str_split(df[i, "title_q"], pattern = " ")), collapse = "+")
            
            begin_date <- df[[i, "date_q"]]
            
            end_date <- df[[i, "date_q"]]
            
            # optional: we just get the url with this parameter
            fl <- 'web_url'
            
            query_url <- paste0("http://api.nytimes.com/svc/search/v2/articlesearch.json?q=", query,
                                "&begin_date=", begin_date,
                                "&end_date=", end_date,
                                #"&fl=", fl,
                                #"&fq=", fq,
                                "&api-key=", nyt_key, 
                                sep = "")
            
            if (verbose == TRUE) {
                print(query_url)
            } else
                
                
                temp <- fromJSON(query_url)
            
            df[[i, "url_api"]] <- temp$response$docs$web_url[1]
            df[[i, "abstract"]]       <- temp$response$docs$abstract[1]
            df[[i, "lead_paragraph"]] <- temp$response$docs$lead_paragraph[1]
            df[[i, "print_section"]]  <- temp$response$docs$print_section[1]
            df[[i, "print_page"]]     <- temp$response$docs$print_page[1]
            df[[i, "headline_main"]]  <- temp$response$docs$headline$main[1]
            df[[i, "headline_print"]]    <- temp$response$docs$headline$print_headline[1]
            
            # fixing the errors when no keywords
            if (length(temp$response$docs$keywords[[1]]) == 0) {
                df[[i, "keywords"]]    <- NA_character_
            } else
                df[[i, "keywords"]]    <- paste(temp$response$docs$keywords[[1]][2])
            
            
            df[[i, "pub_date"]]    <- temp$response$docs$pub_date[1]
            df[[i, "document_type"]]    <- temp$response$docs$document_type[1]
            df[[i, "news_desk"]]    <- temp$response$docs$news_desk[1]
            df[[i, "section_name"]]    <- temp$response$docs$section_name[1]
            
            # fixing the errors when no keywords
            if (length(temp$response$docs$byline[[1]]) == 0) {
                df[[i, "byline"]]    <- NA_character_
            } else 
                df[[i, "byline"]]    <- temp$response$docs$byline[[1]][1]
            
            df[[i, "word_count"]]    <- temp$response$docs$word_count[1]
            df[[i, "api_hits"]]    <- temp$response$meta$hits
            
            print(paste("URL ", df[[i, "id"]], " added;"))
            
            Sys.sleep(10)}, error=function(e){cat("ERROR :",conditionMessage(e), ";\n")})
    }
    return(df)    
}


#  DAY 1 (less than 4k query because of the prior testing that day)
day1 <- nyt_api_query(list_of_articles, 1:3800)

write_csv(day1[1:3800, ], "../data/output/day1_nyt.csv")




# DAY 2 
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day2_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day2 <- nyt_api_query(3801:7800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day2[3801:7800, ], "../data/output/day2_nyt.csv")



# DAY 3 
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day3_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day3 <- nyt_api_query(7801:11800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day3[7801:11800, ], "../data/output/day3_nyt.csv")



# DAY 4 
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day4_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day4 <- nyt_api_query(11801:15800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day4[11801:15800, ], "../data/output/day4_nyt.csv")






# DAY 5 
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day5_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day5 <- nyt_api_query(15801:19800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day5[15801:19800, ], "../data/output/day5_nyt.csv")






# DAY 6 
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day6_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day6 <- nyt_api_query(19801:23800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day6[19801:23800, ], "../data/output/day6_nyt.csv")







# DAY 7
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day7_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day7 <- nyt_api_query(23801:27800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day7[23801:27800, ], "../data/output/day7_nyt.csv")







# DAY 8
# N = 4000
# do the whole thing but logged
sink(file = "../data/output/day8_log.txt")
cat("start of query:")
with_tz(Sys.time(), "CET")


day8 <- nyt_api_query(27801:31800)

cat("end of query:")
with_tz(Sys.time(), "CET")
sink()

write_csv(day8[27801:31800, ], "../data/output/day8_nyt.csv")

## THE FOLLOWING SECTION IS FOR CLEANING OUT API FETCH ERRORS
# your mileage may vary on this, in our version there were some issues with the Sys.sleep initially so there are many timeout and other error that needed to be fixed.

# this section relies on the logs created during the API call.

# temporary measure, reading in results from memory
day1_merge <- day1[1:3800, ]

day2_merge <- day2[3801:7800, ]

day3_merge <- day3[7801:11800, ]

day4_merge <- day4[11801:15800, ]

day5_merge <- day5[15801:19800, ]

day6_merge <- day6[19801:23800, ]

day7_merge <- day7[23801:27800, ]

day8_merge <- day8[27801:31034, ]

# create the dataframe
nyt_api_merge <- bind_rows(day1_merge,
                           day2_merge,
                           day3_merge,
                           day4_merge,
                           day5_merge,
                           day6_merge,
                           day7_merge,
                           day8_merge) %>% 
  mutate(row_nr_temp = row_number())

# my NAs!
na_in_df(nyt_api_merge)



# STEP 1 ----
# all the txt logs
logs <- list.files(path = "./data/output/", full.names = TRUE, pattern="_log")

# importing all log files
logs_show_nothing <- lapply(logs,function(i){
  read_csv(i, skip = 1, n_max = 4000, col_names = "api_logs")
})

# check how many and which http error type
logs_show_nothing_df <- bind_rows(logs_show_nothing) %>%
  mutate(overrun_temp = if_else(str_detect(api_logs, "ERROR : subscript out of bounds"), 1, 0)) %>% 
  filter(overrun_temp == 0) %>% 
  mutate(row_nr_temp = seq(from = 3801, to = 31034),
         error = if_else(str_detect(api_logs, "ERROR :"), 1, 0),
         http_error = if_else(str_detect(api_logs, "ERROR : HTTP error"), 1, 0),
         lenght_zero = if_else(str_detect(api_logs, "length zero"), 1, 0),
         undefined_columns = if_else(str_detect(api_logs, "undefined columns"), 1, 0)) %>% 
  filter(error == 1)
  
logs_show_nothing_df %>% 
  group_by(api_logs) %>% 
  summarise(n_errortype = n())


# source if length_zero error: api cannot find anything
# example api link: 
# http://api.nytimes.com/svc/search/v2/articlesearch.json?q=childcare+solutions+new+world+welfare&begin_date=19970601&end_date=19970601&api-key=your-api-key


nyt_api_fix <- nyt_api_merge %>% 
  filter(is.na(url_api))

# fix first 4k errors
sink(file = "data/output/api_fix1.txt")

nyt_api_fix1 <- nyt_api_query(df = nyt_api_fix, range = 1:4000)

sink()


write_csv(nyt_api_fix1[1:4000, ], "data/output/nyt_api_fix1.csv")




# fix the rest of the errors
# start the query around midnight
sink(file = "data/output/api_fix2.txt")

nyt_api_fix2 <- nyt_api_query(df = nyt_api_fix, range = 4001:6281)

sink()


write_csv(nyt_api_fix2[4001:6281, ], "data/output/nyt_api_fix2.csv")


# combine the two query parts
fix1_complete <- nyt_api_fix1[1:4000, ]
fix2_complete <- nyt_api_fix2[4001:6281, ]

# put them together
nyt_fixed_complete <- bind_rows(fix1_complete, fix2_complete)


# now add the fixed and still NA observations back to the original api merge dataframe
nyt_api_merge2 <- nyt_api_merge %>% 
  filter(!is.na(url_api)) %>% 
  bind_rows(nyt_fixed_complete)


# now fix the last remaining batch
lenght_zero_error <- nyt_api_merge2 %>% 
  filter(is.na(url_api)) %>% 
  mutate(article_year = year(date),
         month_year = month(date),
         year_month = str_c(article_year, month_year, sep = "_"),
         title_long_q = str_to_lower(title))


# using the nyt_trunc_query function to see if the three first word approach is OK
sink(file = "data/output/length_zero_fix.txt")

length_zero_fix <- nyt_trunc_query(df = lenght_zero_error, range = 1:2421, long_title = TRUE)

sink()

write_csv(length_zero_fix, "data/output/length_zero_fix.csv")

na_in_df(length_zero_fix)


# now add it back to the final version
nyt_api_merge3 <- nyt_api_merge2 %>% 
  filter(!is.na(url_api)) %>% 
  bind_rows(length_zero_fix)

nyt_api_corpus <- nyt_api_merge3 %>% 
  select(-ends_with("_q"), -article_year, -month_year, -year_month, -row_nr_temp, -api_hits)

# export corpus
write_csv(nyt_api_corpus, "data/output/nyt_api_corpus.csv")
