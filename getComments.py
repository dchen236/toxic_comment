import praw
import csv
from urllib.error import HTTPError

user_agent = "PythonScript:FIuqRM0YaSnf1g (by u/toxicnlpminer)"
reddit = praw.Reddit(client_id="FIuqRM0YaSnf1g",
                client_secret="NeKfi-iwigdMaQdK-P6nyF-uTdI",
                user_agent=user_agent)


def getSubredditComments(subreddit, numPosts):
    #Get comments for top numPosts submissions
    postList = []
    for submission in reddit.subreddit(subreddit).hot(limit=numPosts):
        postList.append(submission.id)
    
    commentList = []
    for postId in postList:
        print(f"Getting comments for {postId}")
        try:
            submission = reddit.submission(id=postId)
            submission.comments.replace_more(limit=0)
            #Breadth first through comment trees
            for i, comment in enumerate(submission.comments.list()):
                commDict = {
                    "submission_title": submission.title,
                    "subreddit": submission.subreddit.name, 
                    "body": comment.body,
                    "score": comment.score,
                    "permalink": comment.permalink, 
                    "num_replies": len(comment.replies),
                    "time_created": comment.created_utc
                } 
                commentList.append(commDict)
        except e:
            if e.code in [429, 500, 502, 503, 504]:
                print(F"Reddit is down, sleeping...{e.code}")
                time.sleep(60)
                pass
            else:
                raise
        #Write to CSV file
    try:
        print(commentList)
        csv_columns = ["submission_title", "submission_url","subreddit", "body", "score","time_created", "permalink","num_replies"]
        with open(f"data/{subreddit}.csv", 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, delimiter='\t')
            writer.writeheader()
            for data in commentList:
                writer.writerow(data)
    except IOError:
        print("I/O error")

getSubredditComments("changemyview", 10)
getSubredditComments("politics", 10)
getSubredditComments("wholesomememes", 10)
getSubredditComments("askreddit", 10)
