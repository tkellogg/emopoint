import streamlit as st
from train import Model

model = st.sidebar.selectbox('Model', [
    Model("", ""),
    Model("text-embedding-ada-002", "ada-2"),
    Model("text-embedding-3-small", "ada-3-small"),
    Model("text-embedding-3-large", "ada-3-large (min size)", opts={"dimensions": 256}),
    Model("text-embedding-3-large", "ada-3-large (max size)", opts={"dimensions": 3072}),
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