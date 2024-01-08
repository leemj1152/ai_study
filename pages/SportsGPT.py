import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

if not firebase_admin._apps:
    cred = credentials.Certificate("joker-base-firebase-adminsdk-0wo7n-dabec43869.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


def ALL_years_reader():
    ref = db.collection("sports").list_documents()
    for collection in ref:
        for doc in collection.collections():
            for result in doc.stream():
                st.write(result.to_dict().get("event"))


def selected_year_reader(year):
    ref = db.collection("sports").document(year).collections()
    for collection in ref:
        for doc in (
            collection.stream()
            # .where(filter=FieldFilter("event", "==", "EPL"))
            # .where(filter=FieldFilter("gameResult", "==", "홈패"))
        ):
            st.write(doc.to_dict().get("event"))


st.set_page_config(page_title="Sports GPT", page_icon="❓")

st.title("Sports GPT")

with st.sidebar:
    choice_period = st.selectbox("년도", ("ALL", "2023", "2022", "21년"))
    choice_event = st.selectbox(
        "종목",
        ("ALL", "축구", "농구", "야구", "배구"),
        index=None,
        placeholder="종목 선택",
    )
    if choice_event == "축구":
        choice_league = st.selectbox(
            "리그",
            ("ALL", "A리그", "EPL", "세리에", "라리가", "분데스리가", "K리그", "리그앙"),
            index=None,
            placeholder="리그 선택",
        )

# sports의 모든 문서 / 컬렉션 출력
if choice_period == "ALL":
    ALL_years_reader()

else:
    selected_year_reader(choice_period)
