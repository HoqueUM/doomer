import dotenv
import praw
import os

dotenv.load_dotenv()

client = os.getenv("CLIENT_ID")
secret = os.getenv("SECRET")

reddit = praw.Reddit(
    client_id=client,
    client_secret=secret,
    user_agent="doomer/1.0 by u/Your-Simp-Card"
)

collapse = "collapse"
uplifting_news = "UpliftingNews"

subreddit = reddit.subreddit(uplifting_news)

for submission in subreddit.hot(limit=5):
    if not submission.is_self:
        print(submission.title)
        print(submission.url)
