import psycopg2
import logging

from src.conf import app_config as config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(config.log_handler)

def getConnection():
   conn = psycopg2.connect(database="cvasset", user = "postgres", password = "admin", host = "127.0.0.1", port = "5432")
   return conn


def updateUseCase(params, useCaseId):
    if(config.usedb==False): return
    logger.info("updateUseCase")
    conn = getConnection()
    cur = conn.cursor()

    query = "Update usecase SET "
    try:
        for key in params:
            value = params[key]

            query += str(key) + '=' + "'{}'".format(value) + ','
           
        query = query[:-1]
        query += " where usecaseid = '" + str(useCaseId)+"'" #+ "' AND status = 'training_in_progress' "
        cur.execute(query)
        conn.commit()
        print("query = ", query)
    except Exception as e:
        logger.error(e)
        conn.rollback()

    cur.close()
    conn.close()

def checkDB():
    if (config.usedb == False): return
    logger.info("Checking DB")
    conn = psycopg2.connect(database="cvasset", user="postgres", password="admin", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    try:
        query = "Select count(*) from satellite_training_details"
        cur.execute(query)
        data = cursor.fetchone()
        print(data+" Rows found")
        conn.commit()
    except Exception as e:
        logger.error(e)
        conn.rollback()

    cur.close()
    conn.close()
