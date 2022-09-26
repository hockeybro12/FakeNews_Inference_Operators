This README describes our dataset release. 

We release the the initial node representations we used in the graph and some files that enable the graph construction (including metadata such as what users we used, what sources/articles they interacted with, etc.).

### Source, User, Article Representations
We assign each source, user, and article and ID in our graph. In our paper, we computed these representations through a combination of various features. Here, we release those representations as TSV files. Each file contains the Article/Source/User ID we used in our graph (all completely random and unique for each article/source/user), and then the vector that was the representation. The file names are:

article_id_to_representation.tsv
source_id_to_representation.tsv
user_id_to_representation.tsv

Due to file size, we upload these files here:
https://drive.google.com/drive/folders/1idtJmQk0utu7BNGdh03-2wrw5VdHBUoD?usp=sharing


### Other Data

We also provide other data that we had used for constructing the graph. They are provided in this folder:

article_id_to_name.tsv -> This maps the article ID (graph ID) to its name.

article_id_to_source.tsv -> This maps the article ID to its source in the graph.

article_name_to_id.npy -> This maps the article name to its graph ID. 

article_name_to_url.npy -> This maps article names to URLs that they can be downloaded at for the full article headline and context. Not all articles are present yet.

article_user_talkers.npy -> This maps article to the users that propagate them and is used to build the graph.

articles_per_directory.npy -> This maps sources to their articles.

domain_twitter_triplets_dict.npy -> This provides the usernames for the Twitter profiles for sources if they have one.

source_followers_dict.npy -> This provides information on which sources follow which users.

source_name_to_id.tsv -> This maps the source names from the Baly dataset (https://github.com/ramybaly/News-Media-Reliability) to our ID's in the graph. Those ID's were also found in source_id_to_representation.tsv, and allow you to get the source representations.

article_id_to_source.tsv -> This maps the articles to each source ID they came from. All of these ID's are our graph IDs.

user_article_talks.tsv -> For each user (represented by their Twitter ID), this tells you what articles they propagate/interact with. Each is represented by a tuple consisting of the conversation ID (from Twitter, for example the tweet the user made that mentions the article) and the article ID that was interacted with.

user_source_following.tsv -> For each user (represented by their Twitter ID), this tells you what source (represented by the Graph ID) they follow/interact with, and shoould be connected to.

user_twitter_to_id.tsv -> For each user (represented by their Twitter ID), this tells us what is their graph ID.

users_that_follow_each_other.npy -> For each user, what users are they following. This is also in the Google Drive folder.