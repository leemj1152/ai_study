from flask import Flask, render_template, request

from saveFile import save_to_bat_data

app = Flask("DataScrapper")

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/search")
def search():
  episodeNumber = request.args.get("episodeNumber")
  data = save_to_bat_data(episodeNumber)
  return render_template("search.html", data=data, episodeNumber=episodeNumber)

app.run("127.0.0.1")
