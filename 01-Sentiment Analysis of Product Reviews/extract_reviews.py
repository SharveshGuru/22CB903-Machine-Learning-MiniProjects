import requests
from bs4 import BeautifulSoup

# Initialize an empty list to store all reviews
all_reviews = []

# Loop through the first 10 pages of reviews
for page_num in range(1, 432):

    # URL to get the reviews for the specified page
    url_source = 'https://www.flipkart.com/aristocrat-airstop-53-hardbody-trolley-bag-cabin-suitcase-4-wheels-21-inch/product-reviews/itm5a5d80f729676?pid=STCGGDWPMHQRVHZX&lid=LSTSTCGGDWPMHQRVHZXDWCVWS&marketplace=FLIPKART&page={page_num}'
    url = url_source.format(page_num=page_num)
    r = requests.get(url)

    # Extract data using BeautifulSoup
    soup = BeautifulSoup(r.content, 'lxml')

    # Extract reviews using the appropriate HTML tags and classes
    reviews = soup.find_all('p', {'class': "z9E0IG"})
    div_reviews = soup.find_all('div', {'class': "ZmyHeo"})

    # Combine reviews from <p> and <div> tags
    review2 = ''
    for review in reviews:
        review2 += review.text.strip().replace(',', ' ') + '\n'  # Replace commas with spaces

    for div_review in div_reviews:
        review2 += div_review.text.strip().replace(',', ' ') + '\n'  # Replace commas with spaces

    # Add the combined reviews to the list
    all_reviews.append(review2)

# Write all reviews to a CSV file
for review in all_reviews:
    with open('reviews.csv', 'a', encoding='utf-8') as f:
        f.write(review + '\n')
