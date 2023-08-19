from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class DateRangeInput(BaseModel):
    start_date: str
    end_date: str

@app.get("/", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})

@app.post("/predict/")
async def predict_sales(start_date: str = Form(...), end_date: str = Form(...)):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    train_df = pd.read_csv(r'train_data.csv', parse_dates=['date'])
    store_sales = train_df.copy()
    store_sales = store_sales.set_index('date').to_period('D')
    store_sales = store_sales.set_index(['state', 'category_of_product'], append=True)
    average_sales = store_sales.groupby('date').mean()['sales']

    dp = DeterministicProcess(
        index=average_sales.index,
        constant=False,
        order=3,
        drop=True
    )
    X = dp.in_sample()
    
    # Generate the out-of-sample index for the specified date range
    X_fore = dp.out_of_sample(steps=len(pd.date_range(start=start_date, end=end_date)))
    X_fore.index = pd.date_range(start=start_date, end=end_date)

    y = average_sales.copy()
    model = LinearRegression()
    model.fit(X, y)

    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    # Filter the predicted values within the specified date range
    X_fore = X_fore[(X_fore.index >= start_date) & (X_fore.index <= end_date)]
    y_fore = y_fore[(y_fore.index >= start_date) & (y_fore.index <= end_date)]

    # Prepare data for response
    dates = [pred_date.strftime('%Y-%m-%d') for pred_date in X_fore.index]
    predicted_sales = y_fore.tolist()

    # Prepare response
    response = {
        "predictions": [
            {
                "date": dates[i],
                "predicted_sales": predicted_sales[i]
            }
            for i in range(len(dates))
        ]
    }

    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
