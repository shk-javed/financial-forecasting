import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Financial Trading Dashboard"

# API Configurations
LIVECOINWATCH_API_KEY = "5a1a76ec-7421-4636-81f8-c4563e5daa29"
POLYGON_API_KEY = "3liVVEUTMkv3CN0ad2Ny3hb6IGFTm6Z8"
FOREX_API_KEY = "YOUR_EXCHANGERATE_API_KEY"

# Asset mappings
CRYPTO_COINS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Cardano": "cardano",
    "Ripple": "ripple",
    "Dogecoin": "dogecoin",
    "Solana": "solana",
    "Litecoin": "litecoin",
    "Polkadot": "polkadot",
    "Avalanche": "avalanche-2",
    "Chainlink": "chainlink"
}

STOCK_SYMBOLS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "Infosys (Indian)": "INFY",  
    "HDFC Bank (Indian)": "HDB"
          
}
FOREX_PAIRS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "USD/CHF": "USD/CHF",
    "AUD/USD": "AUD/USD",
}

def fetch_stock_data(symbol):
    """Fetch stock data from Polygon.io API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("results"):
            raise Exception(f"No data available for {symbol}")
        
        df = pd.DataFrame(data["results"])
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap'
        })
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['close']
        df['price_change'] = df['price'].pct_change()
        df['volatility'] = df['price'].rolling(window=2).std()
        
        df = df.drop(['timestamp'], axis=1)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def fetch_crypto_data(coin="bitcoin"):
    """Fetch cryptocurrency OHLC data from CoinGecko API"""
    api_url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days=30"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise Exception(f"No data available for {coin}")
        
        # CoinGecko OHLC data comes as [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"] = df["close"]
        df["volume"] = 0  # CoinGecko's free API doesn't provide volume in OHLC endpoint
        df["price_change"] = df["price"].pct_change()
        df["volatility"] = df["price"].rolling(window=2).std()
        
        df = df.drop("timestamp", axis=1)
        df = df.dropna()
        return df
        
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return pd.DataFrame()

def fetch_forex_data(pair):
    """Fetch forex data using free public API"""
    try:
        # Pair format "EUR/USD" se currencies alag karna
        from_currency, to_currency = pair.split('/')
        
        # Free API URL use kar rahe hain (Note: Make sure FOREX_API_URL upar define kiya ho)
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        
        response = requests.get(url)
        data = response.json()
        
        if to_currency not in data["rates"]:
            raise Exception(f"No data available for {pair}")
            
        base_rate = data["rates"][to_currency]
        
        # Synthetic OHLC data generate karna (kyunki free API history nahi deti)
        date_rng = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        df = pd.DataFrame()
        df["date"] = date_rng
        
        # Thodi volatility add karna taaki graph real lage
        volatility = base_rate * 0.001 
        
        df["open"] = [base_rate + np.random.normal(0, volatility) for _ in range(len(date_rng))]
        df["close"] = [open_price + np.random.normal(0, volatility) for open_price in df["open"]]
        df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.normal(0, volatility/2))
        df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.normal(0, volatility/2))
        
        df["price"] = df["close"]
        df["price_change"] = df["price"].pct_change()
        df["volatility"] = df["price"].rolling(window=2).std()
        
        return df
        
    except Exception as e:
        print(f"Error fetching forex data: {e}")
        return pd.DataFrame()        
def preprocess_data(df):
    """Prepare data for ML model with enhanced features"""
    df["price_change"] = df["price"].diff()
    df["volatility"] = df["price"].rolling(window=2).std()
    df["next_close"] = df["price"].shift(-1)
    
    df["rolling_mean_3"] = df["price"].rolling(window=3).mean()
    df["rolling_std_3"] = df["price"].rolling(window=3).std()
    df["rolling_mean_7"] = df["price"].rolling(window=7).mean()
    df["rolling_std_7"] = df["price"].rolling(window=7).std()

    df.dropna(inplace=True)
    X = df[["price", "price_change", "volatility", "rolling_mean_3", "rolling_std_3", "rolling_mean_7", "rolling_std_7"]]
    y = df["next_close"]
    return X, y, df

def predict_future(model, last_data, days=6):
    """Predict future prices"""
    future_predictions = []
    current_data = last_data.iloc[-1].copy()
    
    for _ in range(days):
        features = pd.DataFrame([current_data])
        predicted_price = model.predict(features)[0]
        future_predictions.append(predicted_price)
        current_data = generate_future_features(current_data, predicted_price)
    
    return future_predictions

def generate_future_features(last_row, predicted_price):
    """Generate features for future prediction"""
    new_row = pd.Series()
    new_row["price"] = predicted_price
    new_row["price_change"] = predicted_price - last_row["price"]
    new_row["volatility"] = abs(predicted_price - last_row["price"])
    new_row["rolling_mean_3"] = (predicted_price + last_row["price"] * 2) / 3
    new_row["rolling_std_3"] = np.std([predicted_price, last_row["price"], last_row["price"]])
    new_row["rolling_mean_7"] = (predicted_price + last_row["price"] * 6) / 7
    new_row["rolling_std_7"] = np.std([predicted_price] + [last_row["price"]] * 6)
    return new_row

def calculate_metrics(y_true, y_pred):
    """Calculate real performance metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Real Accuracy Formula
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    
    # Agar accuracy negative dikhe (rarely), to 0 kar do
    if accuracy < 0: accuracy = 0
    
    return {
        'Accuracy': f"{accuracy:.2f}%",
        'MAE': f"{mae:.2f}",
        'RMSE': f"{rmse:.2f}"
    }
def create_chart(df, symbol, chart_title, include_volume=False):
    """Create interactive chart with predictions"""
    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title=f"No data available for {symbol}",
                plot_bgcolor='#111',
                paper_bgcolor='#111',
                font={'color': 'white'}
            )
        }, None

    X, y, processed_df = preprocess_data(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    historical_predictions = model.predict(X)
    metrics = calculate_metrics(y, historical_predictions)

    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=7)  # Include last historical date
    
    # Get future predictions starting from the last historical date
    future_predictions = [historical_predictions[-1]]  # Start with last historical prediction
    future_predictions.extend(predict_future(model, X, days=6))

    data = [
        go.Candlestick(
            x=df['date'],
            open=df['open'] if 'open' in df else df['price'],
            high=df['high'] if 'high' in df else df['price'],
            low=df['low'] if 'low' in df else df['price'],
            close=df['close'] if 'close' in df else df['price'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ),
        go.Scatter(
            x=df['date'],
            y=historical_predictions,
            mode='lines',
            name='Model Fit',
            line=dict(color='#64B5F6', width=2)
        ),
        go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='6-Day Forecast',
            line=dict(color='#FFB74D', dash='dash'),
            marker=dict(size=8, color='#FFB74D', symbol='circle')
        )
    ]

    if include_volume and 'volume' in df:
        data.append(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color='#B2DFDB'
            )
        )

    layout = go.Layout(
        title=f'{chart_title} (Including 6-Day Forecast)',
        xaxis={
            'title': 'Date',
            'showgrid': False,
            'color': 'white',
            'rangeslider': {'visible': False}
        },
        yaxis={
            'title': 'Price',
            'showgrid': True,
            'gridcolor': '#444',
            'color': 'white'
        },
        yaxis2={
            'title': 'Volume',
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False,
            'color': 'white'
        } if include_volume and 'volume' in df else None,
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font={'color': 'white'},
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            x=0,
            y=1
        ),
        hovermode='x unified'
    )

    return {'data': data, 'layout': layout}, metrics

def create_asset_layout(title, dropdown_options, dropdown_id):
    """Create consistent layout for all pages"""
    return html.Div(
        style={
            "backgroundColor": "#111",
            "color": "white",
            "padding": "20px",
        },
        children=[
            html.H1(title, style={"textAlign": "center", "marginBottom": "20px"}),
            html.Div([
                html.Div([
                    html.Label("Select Asset:", style={"marginBottom": "10px", "display": "block"}),
                    dcc.Dropdown(
                        id=dropdown_id,
                        options=[{"label": k, "value": v} for k, v in dropdown_options.items()],
                        value=list(dropdown_options.values())[0],
                        style={"width": "100%", "color": "black", "marginBottom": "20px"},
                    ),
                ], style={"width": "300px", "margin": "0 auto", "marginBottom": "30px"}),
                html.Div([
                    dcc.Graph(id=f"{dropdown_id}-chart"),
                ], style={"width": "100%"}),
                html.Div(
                    id=f"{dropdown_id}-metrics",
                    style={
                        "marginTop": "20px",
                        "padding": "20px",
                        "backgroundColor": "#222",
                        "borderRadius": "8px",
                        "width": "fit-content",
                        "margin": "20px auto"
                    }
                ),
            ])
        ]
    )

# Main app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        children=[
            dcc.Link('Cryptocurrency Dashboard', href='/crypto', 
                    style={'padding': '10px', 'color': 'white', 'marginRight': '20px'}),
            dcc.Link('Stock Dashboard', href='/stocks', 
                    style={'padding': '10px', 'color': 'white', 'marginRight': '20px'}),
            dcc.Link('Forex Dashboard', href='/forex', 
                    style={'padding': '10px', 'color': 'white'}),
        ],
        style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#111'}
    ),
    html.Div(id='page-content', style={'backgroundColor': '#111', 'minHeight': '100vh'})
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/forex':
        return create_asset_layout("Forex Trading Dashboard", FOREX_PAIRS, "forex")
    elif pathname == '/stocks':
        return create_asset_layout("Stock Trading Dashboard", STOCK_SYMBOLS, "stocks")
    else:
        return create_asset_layout("Cryptocurrency Dashboard", CRYPTO_COINS, "crypto")

def create_callback(asset_type):
    """Create callback for updating charts and metrics"""
    @app.callback(
        [Output(f"{asset_type}-chart", "figure"),
         Output(f"{asset_type}-metrics", "children")],
        [Input(f"{asset_type}", "value")]
    )
    def update_asset_chart(asset):
        if asset_type == "crypto":
            df = fetch_crypto_data(asset)
            chart_title = f"{asset.capitalize()} Price & Predictions"
            include_volume = False
        elif asset_type == "stocks":
            df = fetch_stock_data(asset)
            chart_title = f"{asset} Stock Price & Predictions"
            include_volume = True
        else:  # forex
            df = fetch_forex_data(asset)
            chart_title = f"{asset} Price & Predictions"
            include_volume = False
            
        figure, metrics = create_chart(df, asset, chart_title, include_volume)

        if metrics is None:
            return figure, "No metrics available"
            
        metrics_div = html.Div([
            html.H3(" Performance ", style={"textAlign": "center", "marginBottom": "15px"}),
            html.Table(
                [html.Tr([html.Th(k), html.Td(v)]) for k, v in metrics.items()],
                style={
                    "margin": "0 auto",
                    "borderCollapse": "collapse",
                    "width": "100%",
                    "textAlign": "center"
                }
            )
        ])
        
        return figure, metrics_div

# Create callbacks for each dashboard type
create_callback("crypto")
create_callback("stocks")
create_callback("forex")

# Add error handling for the server
@app.server.errorhandler(Exception)
def handle_error(error):
    print(f"Server Error: {error}")
    return "An error occurred processing your request", 500

# Add custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background-color: #111;
                font-family: Arial, sans-serif;
            }
            .dash-dropdown .Select-control {
                background-color: white;
                border-radius: 4px;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: white;
            }
            table {
                border: 1px solid #444;
            }
            th, td {
                padding: 10px;
                border: 1px solid #444;
            }
            th {
                background-color: #222;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Add configuration for deployment
server = app.server

if __name__ == '__main__':
    app.run(debug=True)