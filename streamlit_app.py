import streamlit as st
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import RandomOverSampler
import joblib  
import os 
import altair as alt     


st.set_page_config(
    page_title="Aplikacja do detekcji preferencji pracownika do zmiany zatrudnienia",
    layout="centered", 
    initial_sidebar_state="auto"
)
st.markdown("""
<style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

nazwa_modelu = "regresja_oversampler_kernelpca.joblib"
dane_treningowe = "ankieta_trening.csv"

zamiana_wynagrodzenia = {
    "Od 0 do 2 000 zł": 2000, "Od 2 001 do 4 000 zł": 4000, "Od 4 001 do 6 000 zł": 6000,
    "Od 6 001 do 8 000 zł": 8000, "Od 8 001 do 10 000 zł": 10000, "Od 10 001 do 12 000 zł": 12000,
    "Od 12 001 do 14 000 zł": 14000, "Od 14 001 do 16 000 zł": 16000, "Od 16 001 do 18 000 zł": 18000,
    "Od 18 001 do 20 000 zł": 20000, "Od 20 001 do 22 000 zł": 22000, "Od 22 001 do 24 000 zł": 24000,
    "Od 24 001 do 26 000 zł": 26000, "Od 26 001 do 28 000 zł": 28000, "Od 28 001 do 30 000 zł": 30000,
    "Powyżej 30 001 zł": 32000,
}
opcje_wynagrodzenia = list(zamiana_wynagrodzenia.keys())

zmiana_wyksztalcenie = {
    "Podstawowe": 1, "Gimnazjalne": 2, "Zawodowe": 3,
    "Średnie (w tym policealne)": 4, "Wyższe": 5,
}
opcje_wyksztalcenie = list(zmiana_wyksztalcenie.keys())

zmiana_plec = {"Kobieta": 0, "Mężczyzna": 1}
opcje_plec = list(zmiana_plec.keys())

zamiana_miasto = {
    "Do 5 tys. mieszkańców": 1, "Od 5 do 20 tys. mieszkańców": 2,
    "Od 20 do 100 tys. mieszkańców": 3, "Od 100 do 200 tys. mieszkańców": 4,
    "Powyżej 200 tys. mieszkańców": 5,
}
opcje_miasto = list(zamiana_miasto.keys())


@st.cache_resource #Działa szybciej z cashowaniem modelu
def zaladowanie_modelu():
    pipeline = None
    if os.path.exists(nazwa_modelu):
        try:
            st.info(f"Ładowanie zapisanego modelu z pliku: {nazwa_modelu}")
            pipeline = joblib.load(nazwa_modelu)
            st.success("Model został poprawnie wczytany.")
        except Exception as e:
            st.warning(f"Nie udało się załadować modelu z pliku {nazwa_modelu}. Błąd: {e}. Rozpoczynam trenowanie nowego modelu.")
            pipeline = None 

    if pipeline is None:
        st.warning(f"Plik modelu {nazwa_modelu} nie znaleziony lub wystąpił błąd ładowania. Trenowanie nowego modelu...")

        
        df_trening = pd.read_csv(dane_treningowe)
        st.info(f"Wczytano dane treningowe z {dane_treningowe}")
        X = df_trening.drop('Czy zamierzasz w przeciągu roku zmienić pracodawcę?', axis=1)
        y = df_trening['Czy zamierzasz w przeciągu roku zmienić pracodawcę?']
        pipeline_nowy = Pipeline([
            ('skalowanie', StandardScaler()),
            ('random', RandomOverSampler(
                sampling_strategy='auto',
            )),
            ('pca', KernelPCA(
                n_components=39,
                kernel='rbf',
                gamma=0.013878770818541972
            )),
            ('regresja', LogisticRegression(
                C=1_000_000.0,
                class_weight='balanced',
                max_iter=1000, 
                penalty='l2',
                solver='newton-cholesky',
                tol=0.01,
                warm_start=False
            ))
        ])

        with st.spinner('Trwa trenowanie modelu... To może chwilę potrwać.'):
            pipeline_nowy.fit(X, y)
        st.success("Trenowanie modelu zakończone.")


        joblib.dump(pipeline_nowy, nazwa_modelu)
        st.info(f"Wytrenowany model zapisano do pliku: {nazwa_modelu}")
        pipeline = pipeline_nowy # nowy pipeline... 

    return pipeline


st.image("logo.png", use_container_width=True)
st.title("Aplikacja do detekcji preferencji pracownika do zmiany zatrudnienia")
pipeline = zaladowanie_modelu()

if pipeline is None:
    st.error("Nie można uruchomić aplikacji, ponieważ model nie jest dostępny.")
    st.stop() 

st.header("Ankieta pracownicza")
with st.form("survey_form"):
    plec_opcja = st.selectbox("Podaj płeć", options=opcje_plec)
    wiek = st.number_input("Podaj swój wiek", min_value=18, max_value=75, step=1)
    wyksztalcenie_opcja = st.selectbox("Podaj swoje wyksztalcenie", options=opcje_wyksztalcenie)
    miasto_opcje = st.selectbox("Podaj wielkość swojego miasta/gminy", options=opcje_miasto)
    staz = st.number_input("Ogólny staż zatrudnienia w pełnych latach", min_value=0, max_value=60, step=1)
    wynagrodzenie_opcja = st.selectbox("Podaj swoje wynagrodzenie miesięczne (netto)", options=opcje_wynagrodzenia)
    st.markdown("""
**Oceń w skali:**

1: w pełni się nie zgadzam  
2: umiarkowanie się nie zgadzam  
3: minimalnie się nie zgadzam  
4: minimalnie się zgadzam  
5: umiarkowanie się zgadzam  
6: w pełni się zgadzam
""")
    pytania = [
        "Dobra atmosfera w pracy jest dla mnie bardzo istotna",
        "Wysokie wynagrodzenie stanowi cel główny pracy zarobkowej!",
        "Uważam, że płacą mi odpowiednio za pracę,  którą wykonuję.",
        "W mojej firmie są małe szanse na awans w pracy.",
        "Mój kierownik jest w pełni kompetentny do wykonywania swojej pracy.",
        "Nie jestem zadowolony ze świadczeń, które otrzymuję.",
        "Gdy dobrze wykonuję pracę, jestem odpowiednio doceniany.",
        "Wiele naszych reguł i procedur utrudnia dobre wykonywanie pracy.",
        "Lubię ludzi, z którymi pracuję.",
        "Czasami myślę, ze moja praca jest bez sensu.",
        "Komunikacja w mojej firmie wydaje się dobra.",
        "Podwyżki zdarzają się zbyt rzadko.",
        "Ten, kto dobrze wykonuje swoją pracę ma u nas spore szanse na awans.",
        "Mój kierownik jest wobec mnie niesprawiedliwy.",
        "Świadczenia, które otrzymuję są porównywalne z tymi, które oferuje większość innych firm.",
        "Uważam, że moja praca jest niedoceniana.",
        "Biurokracja rzadko przeszkadza mi w efektywnym wykonywaniu pracy.",
        "Uważam, że muszę pracować ciężej z powodu niekompetencji osób, z którymi pracuję.",
        "Lubię rzeczy, którymi się zajmuję w swojej pracy.",
        "Cele firmy, dla której pracuję, są dla mnie niejasne.",
        "Gdy myślę o swoim wynagrodzeniu, czuję się niedoceniany.",
        "Ludzie awansują w mojej firmie tak samo szybko, jak w innych firmach.",
        "Mój kierownik za mało interesuje się odczuciami podwładnych.",
        "Pakiet świadczeń, które otrzymujemy, jest słuszny.",
        "Jest mało nagród dla tych, którzy tu pracują.",
        "Mam za dużo zadań do wykonania w pracy.",
        "Lubię spędzać czas ze swoimi współpracownikami.",
        "Często mam odczucie, że nie wiem, co się dzieje w mojej firmie.",
        "Jestem dumny ze swojej pracy.",
        "Jestem usatysfakcjonowany perspektywą wzrostu zarobków w przyszłości.",
        "Są dodatkowe świadczenia, których nie otrzymuję, a uważam, że powinienem.",
        "Lubię swojego kierownika.",
        "Mam za dużo papierkowej roboty.",
        "Uważam, że moje wysiłki nie są nagradzane w sposób, w jaki być powinny.",
        "Jestem zadowolony z możliwości awansu.",
        "Moja praca jest przyjemna.",
        "Przydzielane zadania nie są w pełni wyjaśniane."
    ]

    odpowiedzi = {}
    pytania_dla_uzytkownika = [p for p in pytania if p != "Czy zamierzasz w przeciągu roku zmienić pracodawcę?"]
    for q in pytania_dla_uzytkownika:
        odpowiedzi[q] = st.slider(q, 1, 6, 3)

    wyslij = st.form_submit_button("Wyślij i zobacz predykcję")

    if wyslij:
        st.success("Dziękujemy za udział w ankiecie! Za chwilę zobaczysz wyniki predykcji.")
        st.subheader("Podsumowanie ankiety:")


        dane_input = {
            "Podaj płeć": zmiana_plec[plec_opcja],
            "Podaj swój wiek": wiek,
            "Wykształcenie": zmiana_wyksztalcenie[wyksztalcenie_opcja],
            "Podaj wielkość swojego miasta/gminy": zamiana_miasto[miasto_opcje],
            "Ogólny staż zatrudnienia w pełnych latach": staz,
            **odpowiedzi, 
            "Wynagrodzenie": zamiana_wynagrodzenia[wynagrodzenie_opcja],
        }
        df_input = pd.DataFrame([dane_input])


        
        df_trening_cols = pd.read_csv(dane_treningowe, nrows=0) # tylko nagłówki
        X_cols_order = df_trening_cols.drop('Czy zamierzasz w przeciągu roku zmienić pracodawcę?', axis=1).columns
        X_nowe = df_input[X_cols_order]
        with st.expander("Zobacz dane po przetworzeniu"):
            st.dataframe(X_nowe)

        st.divider() 
        try:
            with st.spinner('Dokonywanie predykcji...'):
                y_pred_prob = pipeline.predict_proba(X_nowe)[0]
                y_pred = pipeline.predict(X_nowe)[0]

            col1, col2 = st.columns(2)

            with col1:
                if y_pred == 1:
                    st.warning("**Pracownik prawdopodobnie zamierza zmienić pracę w przeciągu roku.**")
                else:
                    st.success("**Pracownik prawdopodobnie nie zamierza zmieniać pracy w przeciągu roku.**")
                st.text(f"Prawdopodobieństwo pozostania w pracy: {y_pred_prob[0]:.2%}")
                st.text(f"Prawdopodobieństwo zmiany pracy: {y_pred_prob[1]:.2%}")
            with col2:
                st.subheader("Szacowane prawdopodobieństwo:")
                prawdopodobienstwo_df = pd.DataFrame({
                    'Decyzja': ['Pozostanie w pracy', 'Zmiana pracy'],
                    'Prawdopodobieństwo': [y_pred_prob[0], y_pred_prob[1]]
                })
                wykres = alt.Chart(prawdopodobienstwo_df).mark_bar().encode(
                    x=alt.X('Decyzja:N', title=None),
                    y=alt.Y('Prawdopodobieństwo:Q', axis=alt.Axis(format='%')),
                    color=alt.Color('Decyzja:N',
                        scale=alt.Scale(
                            domain=['Pozostanie w pracy', 'Zmiana pracy'],
                            range=['#00CC96', '#EF553B']
                        ),
                        legend=None
                    )
                ).properties(height=300)
                st.altair_chart(wykres, use_container_width=True)



        except Exception as e:
            st.error(f"Wystąpił błąd podczas predykcji: {e}")
            st.exception(e) #tylko debug 
st.divider()
st.markdown("<center><sub>© 2025 Oskar Kałuża - s30217. Aplikacja prezentacyjna do pracy magisterskiej</sub></center>", unsafe_allow_html=True)