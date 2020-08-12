import os

import praw
import requests
from bs4 import BeautifulSoup
from botNet import goodPlace

DOWNLOAD_PATH = "Reddit"

def is_direct_image(url):
    """Checks if a url is a direct image url by checking
    if the url contains a file extension. Only checks
    for png, gif, jpg, and jpeg, as these are the only
    formats we are likely to encounter.
    """
    return any(ex in url.lower() for ex in ['.png', '.jpg', '.jpeg'])


def get_image_url(url):
    """Returns direct image url from imgur page."""
    req = requests.get(url)
    req.raise_for_status()
    soup = BeautifulSoup(req.text, 'html.parser')
    img = soup.find('img', class_='post-image-placeholder')
    try:
        return f'http:{img.get("src")}'
    except AttributeError:
        print(f'Encountered unsupported url: {url}')


def download_image(url, path=DOWNLOAD_PATH, chunksize=512):
    """Downloads each image to the specified download path.
    Uses the image ID as the filename.
    """
    req = requests.get(url)
    req.raise_for_status()
    filename = url.rsplit('/', 1)[-1]
    file = filename.split('.')
    filename = file[0] + ".jpg"
    print(filename)
    with open(os.path.join(path, filename), 'wb') as file:
        for chunk in req.iter_content(chunksize):
            file.write(chunk)
    path1 = os.path.join(path,filename)
    novi = goodPlace(path1)
    os.rename(path1,os.path.join(path,novi))



def download_from_subreddit(sub='TheGoodPlace', sort='hot', lim=100,
                            path=DOWNLOAD_PATH):
    """Downloads images from specifed subreddit."""
    reddit = praw.Reddit(client_id="",
                         client_secret="", password="",
                         user_agent=, username="")

    subreddit = reddit.subreddit(sub)
    subreddit_sort = {
        'hot': subreddit.hot,
        'top': subreddit.top,
        'new': subreddit.new
    }

    for submission in subreddit_sort[sort](limit=lim):
        if submission.stickied or submission.is_self:
            continue
        try:
            url = submission.url
            if not is_direct_image(url):
                url = get_image_url(url)
            if url is not None:
                download_image(url, path=path)
        except:
            continue


if __name__ == '__main__':
    download_from_subreddit()
