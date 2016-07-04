from flask import render_template
from app import app

@app.route("/")
def main():
	item = random.choice(items)
	return item
	items = random.sample(items, 20)
	random_display_items = [] 
	for item in items:
		name , url = item_info[item]
		random_display_items.append(item, name, url)
	data = {}
	data["items"] = random_display_items