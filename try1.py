import streamlit as st

from sqlalchemy import create_engine,text

conn = st.connection('mysql',type='sql')
conn._connect()
query = text('SELECT * FROM df_damage')

with conn.session as s:
    result = s.execute(query)
    st.write(result.fetchall())


