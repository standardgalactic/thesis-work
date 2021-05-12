# from https://www.reddit.com/r/options/comments/9nymxi/tradier_apipython_script_to_download_historical/
#%% load packages
import http
import json
import pandas as pd
import calendar
import os
from datetime import timedelta, date
from typing import Dict, List, Tuple
from http import client
#%%

class Tradier:
    def __init__(self, auth: str, storage_path="data/"):
        self.storage_path = storage_path
        self.connection = http.client.HTTPSConnection(
            "sandbox.tradier.com", 443, timeout=30)
        self.headers = headers = {"Accept": "application/json",
                                  "Authorization": "Bearer {}".format(auth)}

    def request(self, endpoint: str):
        self.connection.request("GET", endpoint, None, self.headers)
        try:
            response = self.connection.getresponse()
            content = response.read()
            if int(str(response.status)) == 200:
                return json.loads(bytes.decode(content))
            return None
        except http.HTTPException as e:
            return e

    def options(self, symbol: str):
        return Options(self, symbol)

    def historical_data(self, symbol: str):
        endpoint = "/v1/markets/history?symbol={}".format(symbol)
        return self.request(endpoint)

    def load_data(self, symbol: str) -> pd.DataFrame:
        path = self.storage_path + "{}.csv".format(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.set_index(pd.DatetimeIndex(df["date"]))
            df = df.loc[:, ~df.columns.str.contains('date')]
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df
        else:
            try:
                df = pd.DataFrame(self.historical_data(
                    symbol).get("history", {}).get("day", []))
                df = df.set_index(pd.DatetimeIndex(df["date"]))
                df = df.loc[:, ~df.columns.str.contains('date')]
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.to_csv(path)
                return df
            except Exception as e:
                print(e)
                return None


def third_fridays(d, n):
    """Given a date, calculates n next third fridays
    https://stackoverflow.com/questions/28680896/how-can-i-get-the-3rd-friday-of-a-month-in-python/28681097"""

    def next_third_friday(d):
        """ Given a third friday find next third friday"""
        d += timedelta(weeks=4)
        return d if d.day >= 15 else d + timedelta(weeks=1)

    # Find closest friday to 15th of month
    s = date(d.year, d.month, 15)
    result = [s + timedelta(days=(calendar.FRIDAY - s.weekday()) % 7)]

    # This month's third friday passed. Find next.
    if result[0] < d:
        result[0] = next_third_friday(result[0])

    for i in range(n - 1):
        result.append(next_third_friday(result[-1]))

    return result


class Options:
    def __init__(self, tradier: Tradier, symbol: str):
        self.api = tradier
        self.symbol = symbol
        self.cache = {}

    def call(self, expiration: date, strike: int) -> pd.DataFrame:
        chain = "{symbol}{y}{m:02d}{d:02d}C{strike:05d}000".format(symbol=self.symbol, y=str(
            expiration.year)[2:], m=expiration.month, d=expiration.day, strike=strike)
        if chain in self.cache:
            return self.cache[chain]
        else:
            df = self.api.load_data(chain)
            self.cache[chain] = df
            return df

    def put(self, expiration: date, strike: int) -> pd.DataFrame:
        chain = "{symbol}{y}{m:02d}{d:02d}P{strike:05d}000".format(symbol=self.symbol, y=str(
            expiration.year)[2:], m=expiration.month, d=expiration.day, strike=strike)
        if chain in self.cache:
            return self.cache[chain]
        else:
            df = self.api.load_data(chain)
            self.cache[chain] = df
            return df

    def initialize_repository(self):
        ''' 
        Download all of the historical price data for all monthly expirations within a 10%
        price range of the underlying for that month. This can be manually changed in the code 
        below. By default, this downloads data from 2018 only. Beyond that is an exercise for the reader
        '''

        # historical price data for the underlying, which we will merge in
        data = self.api.load_data(self.symbol)

        # calculate monthly high and low for the underlying
        monthly_range = {}
        for m in range(1, 13):
        	try:
	            x = data[date(2018, m, 1):date(2018, m + 1, 1)]
	            monthly_range[m] = dict(low=min(x['low']), high=max(x['high']))
        	except:
                # If we don't have data for this month, just extrapolate from the previous month
        		monthly_range[m] = dict(
        		    low=monthly_range[m - 1]['low'], high=monthly_range[m - 1]['high'])

        for m, k in monthly_range.items():
            expiration = third_fridays(date(2018, m, 1), 1)[0]
            # Get all strikes that are 10% below the monthly low and 10% above the monthly high
            strikes = [x for x in range(int(k['low'] * .9), int(k['high'] * 1.1))]
            # Download and save all of the option chains
            for s in strikes:
                self.call(expiration, s)
                self.put(expiration, s)

#%% Call API
api = Tradier("ZG4gAYMDiq0LyEtdboS8F7H007yd")
#print(api.options("SPY").put(date(2018, 9, 21), 175))
api.options("SPY").initialize_repository()