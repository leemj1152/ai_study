from dataScrapper import extract_bet_datas


def save_to_bat_data(episodeNumber):
    data_list = extract_bet_datas(episodeNumber)
    file = open(f"{episodeNumber}data.csv", "w", encoding="utf-8")
    file.write("종목,홈팀,기준점,원정팀,게임유형,승,무,패,최종스코어,경기결과\n")
    for data in data_list:
        file.write(
            f"{data['event']},{data['homeTeam']},{data['standardPoint']},{data['awayTeam']},{data['gameType']},{data['winDistribute']},{data['drawDistribute']},{data['loseDistribute']},{data['finalScore']},{data['gameResult']}\n"
        )
    file.close()
    return data_list
