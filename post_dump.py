import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# === Forum Config ===
cookies = {
    '_forum_session': 'Ns5dHo0vawgJG6cuBfLGmRzWUaIdnvTM7DANmN3%2FO8JbMx48FQHg9ULZ2fxjOjGKm8tvuJVTJ6TMuyaawYrN1d6I0cBfvbmjVMi4%2FKfe7gDvWe0cVfIhRIYLau6Br6XB2nGjhrK6VxNjec88p%2BotdxO0BSED003nThkSOlkkKGeP7awtD%2BVx36Qo1f6hcvZ8AhfmX2rRK1hwwl%2Bamw7esyuKpZFw%2BWQddafWK2UmQ9o88%2F2K%2FfGWOqito2cO4xRE1LGbB0t9zKyr9cVWHcChfYvwzHtAhp8DV5HI0wwLsdiqqBJyrHvWCjeVIj5OlBG9ZUXVYLmPnlzEiz%2BQZMUkvPk3JozWB%2F9ZRE1Amw8vblPtAmvJwnnAx%2BVJryEH3jyxaLrr8xmEHxWMf20sx7H4%2Ba7JGFyoYdUS6ZnoF4FTSDP%2Be6k0hHMK0DS5xT4klHQBZH6EK%2FAs1lngFAG78kpfcLJpcM%2BWYoHzXhV26rMWmBm7nBiAMZX3EUHAtwkNMskl7lfQodAHDr7PLSIBvIrpqVzVj0DN4sLu3pdTrxZEYCKr6g%3D%3D--XtQqENZ9dcF91iow--1eagytZy0p8CnTyr3gN%2BfA%3D%3D',
    '_t': 'xPT4P9CudfZTYedpJK34cfqL5QYPdtL4taU05J529YMYlBIe2irUEsNeygmKxeb0iuij5BUAvkJXVfgPGQ6S8n5vyRKtsSlA1o4uqDoNqU7fVS9eahroJ6ghw8oqKM6bdsiJ4%2FexGjBjzp8bjbgZkT9JyXxNgNpyWvkCpAtq9fCbrjDlpKCzkMSS4%2BTdMF0KY%2B03kPMs0nPia%2FIZLe5SRD6ttOpswcUM3fedHCah67NqyN21naQC4Pi5JRsg18R6c90LrZAB0jSHOAVepHU96vCnPhhZEopuawDbNxf54%2FphtqV8aRpCZBhHn9LX2H7n0gdpeQ%3D%3D--lm6g2GkDJRozEO31--WsHgkJwMvJ%2F4gKsGtOAgnA%3D%3D',
    '_bypass_cache': 'true'
}
headers = {
    'User-Agent': 'Mozilla/5.0'
}
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34

# Define the date range for filtering
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)

# === Forum Scraping Functions ===
def get_topic_ids(category_slug=CATEGORY_SLUG, category_id=CATEGORY_ID):
    topics = []
    for page in range(0, 8):
        url = f"{BASE_URL}/c/{category_slug}/{category_id}.json?page={page}"
        r = requests.get(url, headers=headers, cookies=cookies)
        if r.status_code != 200:
            break
        data = r.json()
        new_topics = data["topic_list"]["topics"]
        if not new_topics:
            break
        topics.extend(new_topics)
    return topics

def get_posts_in_topic(topic_id):
    r = requests.get(f"{BASE_URL}/t/{topic_id}.json", headers=headers, cookies=cookies)
    if r.status_code != 200:
        return []
    data = r.json()
    posts = []
    
    for post in data["post_stream"]["posts"]:
        # Parse the created_at date
        created_at = datetime.fromisoformat(post["created_at"].replace("Z", ""))
        
        # Filter posts based on the date range
        if START_DATE <= created_at <= END_DATE:
            soup = BeautifulSoup(post["cooked"], "html.parser")
            text = soup.get_text()

            # Extract images
            images = []
            for img in soup.find_all('img'):
                img_url = img.get('src')
                if img_url:
                    images.append(img_url)

            # Create a structured post object
            structured_post = {
                "id": post["id"],
                "name": post["name"],
                "username": post["username"],
                "created_at": post["created_at"],
                "cooked": text,  # Use the plain text content
                "images": images,  # List of image URLs
                "post_number": post["post_number"],
                "post_type": post["post_type"],
                "posts_count": post["posts_count"],
                "topic_id": post["topic_id"],
                "topic_slug": post["topic_slug"],
                "post_url": f"{BASE_URL}{post['post_url']}"
            }
            
            posts.append(structured_post)
    
    return posts

# === Dump Posts to JSON File ===
def dump_posts_to_json():
    all_posts = []
    topics = get_topic_ids()
    
    for topic in topics:
        posts = get_posts_in_topic(topic["id"])
        all_posts.extend(posts)

    # Save the structured posts to a JSON file
    with open('posts_dump.json', 'w') as f:
        json.dump({"post_stream": {"posts": all_posts}}, f, indent=4)

# Call the function to dump posts
if __name__ == "__main__":
    dump_posts_to_json()
    print("Posts have been dumped to posts_dump.json.")
