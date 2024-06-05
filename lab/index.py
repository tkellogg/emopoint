import streamlit as st

page = st.sidebar.selectbox("Page", ["Explore", "Train"])
if page == "Explore":
    st.title("Explore")
    import explore
    explore.main()
elif page == "Train":
    st.title("Train")
    import train
    train.main()