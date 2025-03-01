import dotenv
import praw
import os
import pandas as pd

dotenv.load_dotenv()

client = os.getenv("CLIENT_ID")
secret = os.getenv("SECRET")

reddit = praw.Reddit(
    client_id=client,
    client_secret=secret,
    user_agent="doomer/1.0 by u/Your-Simp-Card"
)



def fetch_reddit(subreddit, limit=2000):
    sentiment = ""
    if subreddit == "collapse":
        sentiment = "doomer"
    elif subreddit == "UpliftingNews":
        sentiment = "not doomer"
    else:
        sentiment = "neutral"
    subreddit = reddit.subreddit(subreddit)
    posts = []     
    for submission in subreddit.hot(limit=limit):
        if not submission.is_self:
            posts.append({
                "title": submission.title,
                "sentiment": sentiment
            })
    return posts

collapse = "collapse"
uplifting_news = "UpliftingNews"
moderatepolitics = "moderatepolitics"

data = fetch_reddit(collapse) + fetch_reddit(uplifting_news)

df = pd.DataFrame(data)
df.to_csv("training_data.csv", index=False)