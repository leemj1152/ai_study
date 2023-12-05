from flask import Flask, render_template, request, redirect, send_file

from saveFile import save_to_bat_data
from dataScrapper import extract_bet_datas

app = Flask("DataScrapper")

db = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    episodeNumber = request.args.get("episodeNumber")
    if episodeNumber == None:
        return redirect("/")
    if episodeNumber in db:
        data = db[episodeNumber]
    else:
        data = extract_bet_datas(episodeNumber)
        db[episodeNumber] = data
    return render_template("search.html", data=data, episodeNumber=episodeNumber)


@app.route("/export")
def export():
    episodeNumber = request.args.get("episodeNumber")
    if episodeNumber == None:
        return redirect("/")
    if episodeNumber not in db:
        return redirect(f"/search?episodeNumber={episodeNumber}")
    save_to_bat_data(episodeNumber)
    return send_file(f"{episodeNumber}data.csv", as_attachment=True)


app.run("localhost", debug=True)
