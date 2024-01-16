import datetime
from pydantic import BaseModel
import logging
import pandas as pd
import os
from sqlalchemy import create_engine
from typing import Dict, Any

from fastapi import FastAPI
import uvicorn

from openai import OpenAI
client = OpenAI(api_key=os.environ.get('API_KEY'))

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def get_config() -> str:
    '''Setup parameters for postgres. Returns a data base url string.'''
    # Parameters for the RDS PostgreSQL instance
    PG_HOST = os.environ.get('PG_HOST')
    PG_PORT = os.environ.get('PG_PORT')
    PG_DATABASE = os.environ.get('PG_DATABASE')
    PG_USER = os.environ.get('PG_USER')
    PG_PASSWORD = os.environ.get('PG_PASSWORD')

    # Create the MySQL database connection string
    db_url = f'postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}'
    
    return db_url

class PredictionParams(BaseModel):
    text: str
    password: str

app = FastAPI()

@app.post("/convert")
async def convert(params: PredictionParams) -> Dict[str, Any]:
    '''Endpoint to convert user query in natural language to a structured response using OpenAI API.'''
    today = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    prompt = '''You are a health assistant. I will provide a list of food I ate and you need to sum up and return values in format: 
                Calories, Protein, Carbs, Fats . Use only this format. Return only summed up values in this and nothing else. If you don't know write:
                0, 0, 0, 0 . Approximate if you don't know exact values. Here are the items I ate:'''
    try:
        text = params.text
        password = params.password

        print(today, text)

        completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        )
        print(completion.choices[0].message.content)
        split_values = completion.choices[0].message.content.split(", ")

        data = {
            "Calories": [split_values[0]],
            "Protein": [split_values[1]],
            "Carbs": [split_values[2]],
            "Fats": [split_values[3]],
            "Date": [datetime.datetime.now()]
        }
        df = pd.DataFrame(data)
        db_url = get_config()
        engine = create_engine(db_url)
        df.to_sql('diet', engine, if_exists='append', index=False)

        formatted_string = f"Calories {split_values[0]} kcal, Protein {split_values[1]} g, Carbs {split_values[2]} g, Fats {split_values[3]} g"
        return {"message": "Prediction completed!", "response": formatted_string}
    except Exception as e:
        logging.error(f'Error in /retrain endpoint: {str(e)}')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
