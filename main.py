from flask import Flask, render_template, request, redirect, send_file
from saveFile import save_to_bat_data_csv, save_to_bat_data_json
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
    years = episodeNumber[0:2]
    episode = episodeNumber[-4:]
    return render_template(
        "search.html",
        data=data,
        episodeNumber=episodeNumber,
        years=years,
        episode=episode,
    )


@app.route("/export")
def export():
    episodeNumber = request.args.get("episodeNumber")
    if episodeNumber == None:
        return redirect("/")
    if episodeNumber not in db:
        return redirect(f"/search?episodeNumber={episodeNumber}")
    save_to_bat_data_json(episodeNumber)
    return send_file(f"{episodeNumber}.json", as_attachment=True)


@app.route("/statistics")
def statistics():
    condition = request.args.get("condition")
    # if condition == None:
    #     return redirect("/statistics")
    # database()
    return render_template("statistics.html")


app.run("localhost", debug=True)
