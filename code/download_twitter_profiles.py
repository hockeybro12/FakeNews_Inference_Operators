
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm
import numpy as np

# this is a sample script to download Twitter profiles given IDs 

# load in the Twitter IDs you want to download to profiles_to_download, for example from user_twitter_to_id
# right now it's empty, so the script won't do anything

profiles_to_download = []
for given_username in tqdm(profiles_to_download):

    scraper = sntwitter.TwitterUserScraper(given_username)

    try:
        # if we were able to download the profile
        if scraper.entity:
            # save all the metadata
            profiles_dict = {}

            profiles_dict['bio'] = str(scraper.entity.description)
            profiles_dict['follower_count'] = str(scraper.entity.followersCount)
            profiles_dict['following_count'] = str(scraper.entity.friendsCount)
            profiles_dict['link_count'] = str(scraper.entity.favouritesCount)
            profiles_dict['tweets_count'] = str(scraper.entity.statusesCount)
            profiles_dict['is_verified'] = 0 if str(scraper.entity.verified) == "False" else 1
            profiles_dict['id'] = str(scraper.entity.id)
            profiles_dict['displayname'] = str(scraper.entity.displayname)
            profiles_dict['listedCount'] = str(scraper.entity.listedCount)
            profiles_dict['mediaCount'] = str(scraper.entity.mediaCount)
            profiles_dict['location'] = str(scraper.entity.location)
            profiles_dict['protected'] = str(scraper.entity.protected)
            profiles_dict['profileImageUrl'] = str(scraper.entity.profileImageUrl)
            profiles_dict['profileBannerUrl'] = str(scraper.entity.profileBannerUrl)
            try:
                profiles_dict['created_date'] = scraper.entity.created.strftime('%m_%d_%Y_00_00_00')
            except Exception as e:
                print("Exception determining the created_date")
                print(e)
                profiles_dict['created_date'] = None
            # save in the profiles directory
            np.save('profiles/' + str(given_username.lower()) + '.npy', np.asarray(dict(profiles_dict)))
    except Exception as e:
        print("Exception at user " + str(given_username))
        print(e)

