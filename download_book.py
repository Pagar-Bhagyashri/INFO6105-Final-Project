import requests

# Download A Christmas Carol
url = "https://www.gutenberg.org/files/46/46-0.txt"
response = requests.get(url)

# Save to file
with open("christmas_carol.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Book downloaded successfully!")