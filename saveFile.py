from dataScrapper import extract_bet_datas

def save_to_bat_data(episodeNumber):
  data_list = extract_bet_datas(episodeNumber)
  file = open(f"{episodeNumber}data.csv", "w", encoding="utf-8")
  file.write("event,homeTeam,standardPoint,awayTeam,gameType,winDistribute,drawDistribute,loseDistribute,finalScore,gameResult\n")
  for data in data_list:
    file.write(f"{data['event']},{data['homeTeam']},{data['standardPoint']},{data['awayTeam']},{data['gameType']},{data['winDistribute']},{data['drawDistribute']},{data['loseDistribute']},{data['finalScore']},{data['gameResult']}\n")
  file.close()
  return data_list