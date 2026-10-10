import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="¿Sobrevivirías al Titanic?", page_icon="🚢")


@st.cache_resource          # carga el modelo una sola vez, no en cada clic
def cargar_modelo():
    with open("modelo_titanic.pkl", "rb") as archivo:
        return pickle.load(archivo)


modelo = cargar_modelo()

st.title("🚢 ¿Sobrevivirías al Titanic?")
st.write("Ingresa los datos de un pasajero y un modelo de Machine Learning "
         "estimará su probabilidad de sobrevivir.")

col1, col2 = st.columns(2)

with col1:
    sexo = st.radio("Sexo", ["female", "male"], horizontal=True,
                    format_func=lambda s: "Mujer" if s == "female" else "Hombre")
    edad = st.slider("Edad", 0, 80, 25)
    titulo = st.selectbox("Título", ["Mr", "Mrs", "Miss", "Master", "Rare"],
                          help="Mr = señor · Mrs = señora · Miss = señorita · Master = niño · Rare = otros (Dr, Rev…)")
    clase = st.selectbox("Clase del boleto", [1, 2, 3], index=2,
                         format_func=lambda c: f"{c}ª clase")

with col2:
    tarifa = st.number_input("Tarifa pagada", min_value=0.0, max_value=520.0, value=15.0, step=1.0,
                             help="Referencia: 3ª clase ≈ 8 · 2ª ≈ 20 · 1ª ≈ 85")
    hermanos = st.number_input("Hermanos/as o cónyuge a bordo", 0, 8, 0)
    padres_hijos = st.number_input("Padres o hijos a bordo", 0, 6, 0)
    puerto = st.selectbox("Puerto de embarque", ["S", "C", "Q"],
                          format_func=lambda p: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[p])
    tiene_cabina = st.checkbox("Tenía cabina registrada")

if st.button("🔮 Predecir"):
    pasajero = pd.DataFrame([{
        "Pclass": clase, "Sex": sexo, "Age": edad, "SibSp": hermanos, "Parch": padres_hijos,
        "Fare": tarifa, "Embarked": puerto, "Title": titulo, "HasCabin": int(tiene_cabina),
    }])
    prob = modelo.predict_proba(pasajero)[0, 1]

    st.metric("Probabilidad de sobrevivir", f"{prob:.0%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.success("🛟 ¡Sobrevivirías!")
        st.balloons()
    else:
        st.error("🌊 Lo siento… probablemente no sobrevivirías.")
