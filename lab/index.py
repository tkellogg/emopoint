import streamlit as st
import models

model = st.sidebar.selectbox('Model', [
    models.Model("", ""),
    *models.MODELS,
], format_func=lambda model: model.label)

if not model or not model.id:
    st.stop()

page = st.sidebar.selectbox("Page", ["Explore", "Train", "Sketch", "Text Generation", "Cluster Eval"])
if page == "Explore":
    st.title("Explore")
    import explore
    explore.main()
elif page == "Train":
    st.title("Train")
    import train
    train.main(model)
elif page == "Sketch":
    st.title("Sketch")
    import sketch
    sketch.main()
elif page == "Text Generation":
    st.title("Text Generation")
    import text_gen
    text_gen.main(model)
elif page == "Cluster Eval":
    st.title("Cluster Evaluation")
    import cluster_eval
    cluster_eval.main(model)