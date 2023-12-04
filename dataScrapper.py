from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def extract_bet_datas(episodeNumber):
    browser = webdriver.Chrome()
    browser.get(f"https://www.betman.co.kr/main/mainPage/gamebuy/winrstDetl.do?gmId=G101&gmTs={episodeNumber}&sbx_gmCase=PPT&sbx_gmType=G101&ica_fromDt=2023.09.03&ica_endDt=2023.12.03&rdo=month3&curPage=1&perPage=10")

    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="footer"]')))
    soup = BeautifulSoup(browser.page_source, "html.parser")
    table = soup.find_all('tr', role="row")
    game_results = []
    for games in table:
        game = games.find_all('td')
        if len(game) != 0: # true
            gameNum, kind, date, resultDate, gameType_span, vsTeam, winDistribute, drawDistribute, loseDistribute, score, gameResult = games.find_all('td')
            sportKind = kind.find('span', class_="db")
            vsTeambox = vsTeam.find('div', class_="vsDIv")
            homeTeam = vsTeambox.find_all('div')[0]
            handyCheck = homeTeam.find('span')
            gameType = gameType_span.find('span', class_="badge")
            awayTeam = vsTeambox.find_all('div')[-1]
            awayTeam.find('span').decompose()
            winDistribute.find('br').decompose()
            loseDistribute.find('br').decompose()
            if drawDistribute.find('br'):
                drawDistribute.find('br').decompose()
            if handyCheck != None:
                handy = handyCheck.string
                homeTeam.find('span').decompose()
                game_result = {
                    'event':sportKind.string,
                    'homeTeam':homeTeam.string,
                    'standardPoint':handy,
                    'awayTeam':awayTeam.string,
                    'gameType':gameType.string,
                    'winDistribute':winDistribute.text,
                    'drawDistribute':drawDistribute.text,
                    'loseDistribute':loseDistribute.text,
                    'finalScore':score.string,
                    'gameResult':gameResult.string,
                }
                game_results.append(game_result)
                # print(f"홈팀 : {homeTeam.string}, 핸디캡/언오버 : {handy}, 원정팀 :{awayTeam.string}, 게임 유형 : {gameType.string}, 홈팀 승리 배당 : {winDistribute.text}, 무승부 배당 : {drawDistribute.text}, 홈팀 패배 배당 : {loseDistribute.text}, 최종 스코어 : {score.string}, 결과 : {gameResult.string}")
            elif handyCheck == None:
                game_result = {
                    'event':sportKind.string,
                    'homeTeam':homeTeam.string,
                    'standardPoint':"",
                    'awayTeam':awayTeam.string,
                    'gameType':gameType.string,
                    'winDistribute':winDistribute.text,
                    'drawDistribute':drawDistribute.text,
                    'loseDistribute':loseDistribute.text,
                    'finalScore':score.string,
                    'gameResult':gameResult.string,
                }
                game_results.append(game_result)
                # print(f"홈팀 : {homeTeam.string}, 원정팀 :{awayTeam.string}, 게임 유형 : {gameType.string}, 홈팀 승리 배당 : {winDistribute.text}, 무승부 배당 : {drawDistribute.text}, 홈팀 패배 배당 : {loseDistribute.text}, 최종 스코어 : {score.string}, 결과 : {gameResult.string}")
    return game_results
