"""
1) Write your auth token in __init__
2) Example usage:
save_to = "C:/options/SPY/{:s}.csv" # "{:s}" becomes option symbol
symbol = "SPY"
start_date = datetime(2015, 1, 8)
end_date = datetime(2021, 12, 31)
download_tradier_options(symbol, start_date, end_date, save_to)

3) optional parameters:
download_tradier_options(symbol, start_date, end_date, save_to, strike_increments=5, strikes=5)
strikes -> how many strikes to download, computed from the closest ATM price
strike_increments -> distance in points between each strikes
"""

class TradierAPI(object):
    def __init__(self, connection_url="https://sandbox.tradier.com/v1", auth_token="ZG4gAYMDiq0LyEtdboS8F7H007yd"):
        self.connection_url = connection_url
        self.headers = headers = {"Accept": "application/json",
                                  "Authorization": "Bearer {}".format(auth_token)}
    def request(self, url, params={}, format="dataframe"):
        status_code = 0
        while (status_code != 200):
            r = requests.get("{:s}/{:s}".format(self.connection_url, url), params=params, headers=self.headers)
            print("Querying '{:s}'".format(r.url))
            status_code = r.status_code
            if (status_code != 200):
                print("REQUEST FAILED. Timing out for 60 seconds.")
                time.sleep(60)
        try:
            result = r.json()
        except:
            return r

        if (format == "dataframe" and result['history'] is None):
            result = pd.DataFrame()
        elif format == "dataframe" and result['history'] is not None:
            try:
                result = pd.DataFrame(result['history']['day'])
                result = result.set_index(pd.DatetimeIndex(result['date']))
            except:
                result = pd.DataFrame()
        
        return result

def DownloadTradierOptions(symbol, start_date, end_date, csv_folder, strike_increments=1, strikes=15):
    api = TradierAPI()
    history_date = start_date - timedelta(days=600)
    history = api.request("markets/history", {"symbol":symbol, "start":history_date.strftime("%y%m%d"), "interval":"monthly"})

    while (start_date < end_date):
        check_date_start = start_date - timedelta(days=3650)
        #pick the closest strike
        monthly_price = int(history.iloc[history.index.get_loc(pd.to_datetime(start_date), method='nearest')]['close'])
        option_symbol = "{:s}{:s}P{:05}000".format(symbol, start_date.strftime("%y%m%d"), monthly_price)
        print(option_symbol)
        option_history = api.request("markets/history", {"symbol": option_symbol, "start":check_date_start.strftime("%Y-%m-%d"), "end":datetime.now().strftime("%Y-%m-%d")})
        if not option_history.empty:
            print("{:s} is a valid expiration date. Downloading {:d} strikes with increments {:d}".format(start_date.strftime("%y%m%d"), strikes, strike_increments))
            #Expiry date exists, grab more strikes for puts and calls
            for option_type in ["C", "P"]:
                for strike_price in range(monthly_price-(strikes*strike_increments), monthly_price+(strikes*strike_increments), strike_increments):
                    
                    option_symbol = "{:s}{:s}{:s}{:05}000".format(symbol, start_date.strftime("%y%m%d"), option_type, int(strike_price))
                    csv_path = csv_folder.format(option_symbol)
                    
                    option_history = api.request("markets/history", {"symbol": option_symbol, "start":check_date_start.strftime("%Y-%m-%d"), "end":datetime.now().strftime("%Y-%m-%d")})
                    if not option_history.empty:
                        print("Saving '{:s}'".format(csv_path))
                        option_history.to_csv(csv_path)

        start_date = start_date + timedelta(days=1)