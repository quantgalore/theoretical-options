# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import math

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from scipy.stats import norm

def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.

    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).

    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")
        
def call_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "call"

    def f_call(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    call_newton_vol = optimize.newton(f_call, x0=0.50, tol=0.05, maxiter=50)
    return call_newton_vol

def put_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "put"

    def f_put(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    put_newton_vol = optimize.newton(f_put, x0=0.50, tol=0.05, maxiter=50)
    return put_newton_vol

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

# =============================================================================
# Prod Calcs.
# =============================================================================

production_dates = calendar.schedule(start_date = (datetime.today()-timedelta(days=10)), end_date = (datetime.today())).index.strftime("%Y-%m-%d").values
full_dates = calendar.schedule(start_date = (datetime.today()-timedelta(days=10)), end_date = (datetime.today()+timedelta(days=10))).index.strftime("%Y-%m-%d").values

# Verify that the trading date is a day that has at least begun trading. If not, replace -1 with -2 to get the session that will have data.
trading_date = production_dates[-1]
next_day = full_dates[full_dates > trading_date][0]

ticker = "C"

# 2 = if the realized move is 2x greater than implied.
move_adjustment = 2

underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{trading_date}/{trading_date}?adjusted=false&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying_data.index = pd.to_datetime(underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
underlying_data = underlying_data[(underlying_data.index.time <= pd.Timestamp("16:00").time())].copy()

price = underlying_data["c"].iloc[-1]

calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=call&as_of={trading_date}&expiration_date.gt={trading_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
calls["days_to_exp"] = (pd.to_datetime(calls["expiration_date"]) - pd.to_datetime(trading_date)).dt.days
calls = calls[calls["days_to_exp"] == calls["days_to_exp"].min()].copy()
exp_date = calls["expiration_date"].iloc[0]

calls["distance_from_price"] = abs(calls["strike_price"] - price)

atm_call = calls.nsmallest(1, "distance_from_price")

atm_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_call['ticker'].iloc[0]}?&order=desc&limit=1000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
atm_call_quotes.index = pd.to_datetime(atm_call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
atm_call_quotes["mid_price"] = round((atm_call_quotes["bid_price"] + atm_call_quotes["ask_price"]) / 2, 2)
atm_call_quotes = atm_call_quotes[atm_call_quotes.index.date <= pd.to_datetime(trading_date).date()].copy()

atm_call_quote = atm_call_quotes.head(1)

puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=put&as_of={trading_date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
puts["distance_from_price"] = abs(price - puts["strike_price"])

atm_put = puts[puts["strike_price"] == atm_call["strike_price"].iloc[0]].copy()

atm_put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_put['ticker'].iloc[0]}?order=desc&limit=1000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
atm_put_quotes.index = pd.to_datetime(atm_put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
atm_put_quotes["mid_price"] = round((atm_put_quotes["bid_price"] + atm_put_quotes["ask_price"]) / 2, 2)
atm_put_quotes = atm_put_quotes[atm_put_quotes.index.date <= pd.to_datetime(trading_date).date()].copy()

atm_put_quote = atm_put_quotes.head(1)

current_time = (pd.to_datetime(trading_date).tz_localize("America/New_York") + timedelta(hours = datetime.now().time().hour, minutes = datetime.now().time().minute))#(pd.to_datetime(trading_date).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp(trade_time).time().hour, minutes = pd.Timestamp(trade_time).time().minute))
time_to_expiration = (((pd.to_datetime(exp_date).tz_localize("America/New_York") + timedelta(hours = 16)) - current_time).total_seconds() / 86400) / 365

call_atm_vol = call_implied_vol(S=price, K=atm_call["strike_price"].iloc[0], t=time_to_expiration, r=.05, option_price=atm_call_quote["mid_price"].iloc[0])
put_atm_vol = put_implied_vol(S=price, K=atm_put["strike_price"].iloc[0], t=time_to_expiration, r=.05, option_price=atm_put_quote["mid_price"].iloc[0])

atm_vol = round(((call_atm_vol + put_atm_vol) / 2)*100, 2)

expected_move = (round((atm_vol / np.sqrt(252)), 2)/100) * move_adjustment

lower_price = round(price - (price * expected_move), 2) - .01
upper_price = round(price + (price * expected_move), 2) + .01

next_day_open = (pd.to_datetime(next_day).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp("09:30").time().hour, minutes = pd.Timestamp("09:30").time().minute))
next_day_close = (pd.to_datetime(next_day).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp("15:50").time().hour, minutes = pd.Timestamp("15:50").time().minute))
next_day_market_close = (pd.to_datetime(next_day).tz_localize("America/New_York") + timedelta(hours = pd.Timestamp("16:00").time().hour, minutes = pd.Timestamp("16:00").time().minute))

tte_at_open = (((pd.to_datetime(exp_date).tz_localize("America/New_York") + timedelta(hours = 16)) - next_day_open).total_seconds() / 86400) / 365
tte_at_close = (((pd.to_datetime(exp_date).tz_localize("America/New_York") + timedelta(hours = 16)) - next_day_close).total_seconds() / 86400) / 365

otm_calls = calls[(calls["strike_price"] >= price)].sort_values("strike_price", ascending=True).head(10)
otm_puts = puts[(puts["strike_price"] <= price)].sort_values("distance_from_price", ascending = True).head(10)

theo_call_list = []

# call_ticker = otm_calls["ticker"].values[0]
for call_ticker in otm_calls["ticker"].values:
    
    try:
    
        call = otm_calls[otm_calls["ticker"] == call_ticker].copy()
        
        call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{call['ticker'].iloc[0]}?&order=desc&limit=50&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        call_quotes.index = pd.to_datetime(call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        call_quotes["mid_price"] = round((call_quotes["bid_price"] + call_quotes["ask_price"]) / 2, 2)
        call_quotes = call_quotes[call_quotes.index.date <= pd.to_datetime(trading_date).date()].copy()
        
        call_quote = call_quotes.head(1)
        
        call_vol = call_implied_vol(S=price, K=call["strike_price"].iloc[0], t=time_to_expiration, r=.05, option_price=call_quote["mid_price"].iloc[0])
        
        call_value_at_open = black_scholes(option_type="call", S=upper_price, K=call["strike_price"].iloc[0], t=tte_at_open, r=.05, q=0, sigma = call_vol)
        call_value_at_close = black_scholes(option_type="call", S=upper_price, K=call["strike_price"].iloc[0], t=tte_at_close, r=.05, q=0, sigma = call_vol)
        
        theo_call_pnl_percent = round(((call_value_at_close - call_quote["mid_price"].iloc[0]) / call_quote["mid_price"].iloc[0])*100, 2)
        
        theo_call_data = pd.DataFrame([{"current_spot_price": price, "theo_future_price": upper_price,
                                        "strike_price": call["strike_price"].iloc[0],
                                        "theo_pnl_pct": theo_call_pnl_percent,"ticker": call_ticker, "side": "call",
                                        "date": trading_date}])
        
        theo_call_list.append(theo_call_data)
        
    except Exception as data_error:
        continue
    
full_theo_call_data = pd.concat(theo_call_list)

#

theo_put_list = []

# put_ticker = otm_puts["ticker"].values[0]
for put_ticker in otm_puts["ticker"].values:
    
    try:
    
        put = otm_puts[otm_puts["ticker"] == put_ticker].copy()
        
        put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{put['ticker'].iloc[0]}?&order=desc&limit=50&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        put_quotes.index = pd.to_datetime(put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        put_quotes["mid_price"] = round((put_quotes["bid_price"] + put_quotes["ask_price"]) / 2, 2)
        put_quotes = put_quotes[put_quotes.index.date <= pd.to_datetime(trading_date).date()].copy()
        
        put_quote = put_quotes.head(1)
        
        put_vol = put_implied_vol(S=price, K=put["strike_price"].iloc[0], t=time_to_expiration, r=.05, option_price=put_quote["mid_price"].iloc[0])
        
        put_value_at_open = black_scholes(option_type="put", S=lower_price, K=put["strike_price"].iloc[0], t=tte_at_open, r=.05, q=0, sigma = put_vol)
        put_value_at_close = black_scholes(option_type="put", S=lower_price, K=put["strike_price"].iloc[0], t=tte_at_close, r=.05, q=0, sigma = put_vol)
        
        theo_put_pnl_percent = round(((put_value_at_close - put_quote["mid_price"].iloc[0]) / put_quote["mid_price"].iloc[0])*100, 2)
        
        theo_put_data = pd.DataFrame([{"current_spot_price": price, "theo_future_price": lower_price,
                                        "strike_price": put["strike_price"].iloc[0],
                                        "theo_pnl_pct": theo_put_pnl_percent,"ticker": put_ticker, "side": "put",
                                        "date": trading_date}])
        
        theo_put_list.append(theo_put_data)
        
    except Exception as data_error:
        continue
    
full_theo_put_data = pd.concat(theo_put_list)

######

full_options_data = pd.concat([full_theo_call_data, full_theo_put_data], axis = 0).sort_values(by="strike_price", ascending=False)
