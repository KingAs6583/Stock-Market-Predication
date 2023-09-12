from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean


SQLALCHEMY_DB_URL = "sqlite:///./Users.db"

engine = create_engine(SQLALCHEMY_DB_URL, connect_args={
                       "check_same_thread": False})
SessionLocal = sessionmaker(autocommit=True, bind=engine)

Base = declarative_base()
session = SessionLocal()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close


# Model
class User(Base):
    __tablename__ = "users"
    user_id = Column(String(255), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), index=True, nullable=False)


# creating database
Base.metadata.create_all(bind=engine)

'''
https://www1.nseindia.com/products/content/equities/indices/indices.htm
https://www.niftyindices.com/reports/historical-data
{% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for message in messages %} 
        <p>{{message}}</p>
      {% endfor %}
    {% endif %}
  {% endwith %}

 {% with messages = get_flashed_messages() %}
  {% if messages %}
  <script>
    var messages = "{{ messages | safe }}";
      alert(messages);
  </script>
  {% endif %}
  {% endwith %}
https://colorlib.com/wp/bootstrap-file-uploads/
https://codepen.io/Matty1515/pen/OYJeoV     
https://codepen.io/kbocz/pen/vbBEBN
'''
