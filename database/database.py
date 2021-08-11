import os

from sqlalchemy import *
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

POSTGRES_USER = 'postgres'
POSTGRES_PW = 'postgresql'
POSTGRES_URL = '192.168.8.100:5432'
POSTGRES_DB = 'mestrado'

# database connection
engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True, pool_pre_ping=True)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

class EnergyConsumption(Base):
    __tablename__ = 'energy_consumption'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True)
    energy = Column(Float, nullable=False)

    def __init__(self, date, energy):
        self.date = date
        self.energy = energy

class EnergyConsumptionForecast(Base):
    __tablename__ = 'energy_consumption_forecast'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True)
    energy = Column(Float, nullable=False)

    def __init__(self, date, energy):
        self.date = date
        self.energy = energy

def add_energy_consumption(date, energy):
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        row = EnergyConsumption(date, energy)
        db_session.add(row)
        db_session.commit()
        db_session.close()
        return 1

    except Exception as e:
        print(e)
        return -1

def add_energy_consumption_forecast(date, energy):
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        row = EnergyConsumptionForecast(date, energy)
        db_session.add(row)
        db_session.commit()
        db_session.close()
        return 1

    except Exception as e:
        print(e)
        return -1

'''
    return 0: zero results
    return -1: error
'''

def get_last_timestamp():
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        response = db_session.query(EnergyConsumption).order_by(EnergyConsumption.date.desc()).limit(1).all()
        db_session.close()
        
        for i in response:
            return response[0].date

        return 0

    except Exception as e:
        print(e)
        return -1

def get_last_timestamp_forecast():
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        response = db_session.query(EnergyConsumptionForecast).order_by(EnergyConsumptionForecast.date.desc()).limit(1).all()
        db_session.close()
        
        for i in response:
            return response[0].date

        return 0

    except Exception as e:
        print(e)
        return -1

'''
    return 1: OK
    return -1: error
'''
'''
def add_last_id(domain, device, last_id):
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        row = ExatronicLastIDs(domain, device, last_id)
        db_session.add(row)
        db_session.commit()
        db_session.close()
        return 1

    except Exception as e:
        print(e)
        return -1

def update_last_id(domain, device, last_id):
    engine = create_engine('postgresql+psycopg2://' + POSTGRES_USER + ':' + POSTGRES_PW + '@' + POSTGRES_URL + '/' + POSTGRES_DB, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    try:
        db_session.query(ExatronicLastIDs.last_id).filter_by(domain = domain, device = device).update({"last_id": (last_id)})
        db_session.commit()
        db_session.close()
        return 1

    except Exception as e:
        print(e)
        return -1
'''
# create tables if not exist
Base.metadata.create_all(bind=engine)