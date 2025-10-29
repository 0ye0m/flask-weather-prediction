from flask import Flask, render_template, request
import requests
import pandas as pd
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY") or "8ff9c20a5352876adddd1b768cdbc2c9"

app = Flask(__name__)
matplotlib.use("Agg")


def get_current_weather(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=10)
    try:
        data = r.json()
    except ValueError:
        return None, {"error": "Invalid response from weather API."}
    if r.status_code != 200 or data.get("cod") not in (200, "200"):
        return None, data
    return data, None


def get_hourly_by_onecall(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    r = requests.get("https://api.openweathermap.org/data/2.5/onecall", params=params, timeout=10)
    try:
        data = r.json()
    except ValueError:
        return None, {"error": "Invalid response from onecall API."}
    if r.status_code == 200 and "hourly" in data:
        return data["hourly"], None
    return None, data


def get_hourly_from_forecast(lat, lon, limit=48):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    r = requests.get("https://api.openweathermap.org/data/2.5/forecast", params=params, timeout=10)
    try:
        data = r.json()
    except ValueError:
        return None, {"error": "Invalid response from forecast API."}
    if r.status_code != 200 or "list" not in data:
        return None, data
    hourly = []
    for item in data["list"]:
        hourly.append(item)
        if len(hourly) >= limit:
            break
    return hourly, None


@app.route("/", methods=["GET", "POST"])
def home():
    search_done = False
    if request.method == "POST":
        city = request.form.get("city", "").strip()
        if not city:
            return render_template("404_error.html", message="Please enter a city name.")
        data, err = get_current_weather(city)
        if err:
            msg = err.get("message", "City not found or invalid API key.")
            return render_template("404_error.html", message=msg)
        city_name = data.get("name")
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        current_temp = round(main.get("temp", 0))
        feels_like = round(main.get("feels_like", 0))
        temp_min = round(main.get("temp_min", 0))
        temp_max = round(main.get("temp_max", 0))
        humidity = round(main.get("humidity", 0))
        country = data.get("sys", {}).get("country", "")
        description = weather.get("description", "")
        search_done = True
        return render_template(
            "index.html",
            city=city_name,
            current_temp=current_temp,
            temp_max=temp_max,
            temp_min=temp_min,
            description=description,
            feels_like=feels_like,
            country=country,
            status=search_done,
            humidity=humidity,
        )
    return render_template("index.html", status=search_done)


@app.route("/predict-weather", methods=["GET", "POST"])
def prediction():
    predict_status = False
    if request.method == "POST":
        city = request.form.get("city", "").strip()
        if not city:
            return render_template("404_error.html", message="Please enter a city name.")
        current, err = get_current_weather(city)
        if err:
            msg = err.get("message", "City not found or invalid API key.")
            return render_template("404_error.html", message=msg)
        coord = current.get("coord", {})
        lat = coord.get("lat")
        lon = coord.get("lon")
        if lat is None or lon is None:
            return render_template("404_error.html", message="Coordinates not found for city.")

        hourly, err = get_hourly_by_onecall(lat, lon)
        if err or not hourly:
            hourly, err = get_hourly_from_forecast(lat, lon, limit=48)
            if err or not hourly:
                msg = (
                    err.get("message", "Hourly forecast not available from APIs.")
                    if isinstance(err, dict)
                    else "Hourly forecast not available."
                )
                return render_template("404_error.html", message=msg)

        temperature = []
        humidity = []
        hours = []
        for i, item in enumerate(hourly[:48]):
            if "temp" in item:
                temp_val = item["temp"]
            elif "main" in item and "temp" in item["main"]:
                temp_val = item["main"]["temp"]
            else:
                temp_val = None
            hum_val = item.get("humidity") or (item.get("main", {}).get("humidity"))
            if temp_val is None or hum_val is None:
                continue
            temperature.append(float(temp_val))
            humidity.append(float(hum_val))
            hours.append(i)

        if len(temperature) < 10:
            return render_template("404_error.html", message="Not enough hourly data for prediction.")

        dict_data = {"hours": hours[::-1], "temp": temperature, "hum": humidity}
        df = pd.DataFrame(dict_data)
        os.makedirs("static/csv", exist_ok=True)
        df.to_csv("static/csv/weather_data.csv", index=False)

        data = pd.read_csv("static/csv/weather_data.csv", index_col="hours").dropna()
        weather_data = data["temp"]
        hum_data = data["hum"]

        warnings.filterwarnings("ignore")
        weather_fit = auto_arima(weather_data, trace=False, suppress_warnings=True, error_action="ignore")
        weather_param = weather_fit.get_params().get("order", (1, 0, 0))
        hum_fit = auto_arima(hum_data, trace=False, suppress_warnings=True, error_action="ignore")
        hum_param = hum_fit.get_params().get("order", (1, 0, 0))

        model_temp = ARIMA(weather_data, order=weather_param)
        model_temp_fit = model_temp.fit()
        model_hum = ARIMA(hum_data, order=hum_param)
        model_hum_fit = model_hum.fit()

        future_count = 5
        start_idx = len(weather_data)
        end_idx = start_idx + future_count - 1

        index_future_time = [datetime.now() + timedelta(hours=i) for i in range(future_count)]
        s_index_future_hours = [t.strftime("%H:%M") for t in index_future_time]

        weather_pred = model_temp_fit.predict(start=start_idx, end=end_idx, typ="levels")
        weather_pred.index = s_index_future_hours
        list_file = weather_pred.to_list()
        temperature_1, temperature_2, temperature_3, temperature_4, temperature_5 = [
            round(x, 1) for x in list_file
        ]

        hum_pred = model_hum_fit.predict(start=start_idx, end=end_idx, typ="levels")
        hum_pred.index = s_index_future_hours
        list_file2 = hum_pred.to_list()
        humidity_1, humidity_2, humidity_3, humidity_4, humidity_5 = [round(x, 1) for x in list_file2]

        city_name = current.get("name")
        main = current.get("main", {})
        weather = current.get("weather", [{}])[0]
        current_temp = round(main.get("temp", 0))
        feels_like = round(main.get("feels_like", 0), 1)
        temp_min = round(main.get("temp_min", 0), 1)
        temp_max = round(main.get("temp_max", 0), 1)
        humidity_now = round(main.get("humidity", 0), 1)
        country = current.get("sys", {}).get("country", "")
        description = weather.get("description", "")
        predict_status = True
        search_done = True

        graph_temp = list(zip(s_index_future_hours, [temperature_1, temperature_2, temperature_3, temperature_4, temperature_5]))
        tlabels = [r[0] for r in graph_temp]
        tvalues = [r[1] for r in graph_temp]

        graph_hum = list(zip(s_index_future_hours, [humidity_1, humidity_2, humidity_3, humidity_4, humidity_5]))
        hlabels = [r[0] for r in graph_hum]
        hvalues = [r[1] for r in graph_hum]

        return render_template(
            "index.html",
            predicted_temp=weather_pred,
            predicted_humidity=hum_pred,
            predict_status=predict_status,
            status=search_done,
            temperature_1=temperature_1,
            temperature_2=temperature_2,
            temperature_3=temperature_3,
            temperature_4=temperature_4,
            temperature_5=temperature_5,
            humidity_1=humidity_1,
            humidity_2=humidity_2,
            humidity_3=humidity_3,
            humidity_4=humidity_4,
            humidity_5=humidity_5,
            city=city_name,
            current_temp=current_temp,
            temp_max=temp_max,
            temp_min=temp_min,
            description=description,
            feels_like=feels_like,
            country=country,
            humidity=humidity_now,
            tlabels=tlabels,
            tvalues=tvalues,
            hlabels=hlabels,
            hvalues=hvalues,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
