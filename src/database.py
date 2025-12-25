import sqlite3
import datetime
import os

DB_NAME = "predictions.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            credit_score INTEGER,
            age INTEGER,
            tenure INTEGER,
            balance REAL,
            products_number INTEGER,
            credit_card INTEGER,
            active_member INTEGER,
            estimated_salary REAL,
            product_price REAL,
            prediction INTEGER,
            churn_probability REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized.")

def save_prediction(input_data: dict, prediction: int, churn_prob: float):
    conn = get_db_connection()
    c = conn.cursor()
    
    # Add timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    c.execute('''
        INSERT INTO predictions (
            timestamp, credit_score, age, tenure, balance, 
            products_number, credit_card, active_member, estimated_salary, 
            product_price, prediction, churn_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        input_data['credit_score'],
        input_data['age'],
        input_data['tenure'],
        input_data['balance'],
        input_data['products_number'],
        input_data['credit_card'],
        input_data['active_member'],
        input_data['estimated_salary'],
        input_data['product_price'],
        prediction,
        churn_prob
    ))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=10):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    
    # Convert rows to dicts
    return [dict(row) for row in rows]
