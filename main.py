# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st

st.title('计数器示例')
count = 0

increment = st.button('递增')
if increment:
    count += 1

st.write('计数 = ', count)
