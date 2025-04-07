import logging
from typing import Tuple, Any
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import sqlite3
import os
import time  # Import time module to add a delay if needed
from PIL import Image # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Investment Dashboard", layout="wide")
st.title("📈 Multi-Asset Investment Dashboard")

# Load and display the image
try:
    # Try loading from GitHub first
    github_url = "https://raw.githubusercontent.com/segzee/AKH_PROJECT/master/download.png"
    try:
        from urllib.request import urlopen
        from io import BytesIO
        response = urlopen(github_url)
        img = Image.open(BytesIO(response.read()))
        st.image(img, width=200, caption="Let Invest")
    except Exception as url_error:
        # Fallback to local file if GitHub fails
        image_path = r"c:\Users\Segun\Desktop\AkH Projects\download.png"
        img = Image.open(image_path)
        st.image(img, width=200, caption="AkH Capital Logo")
except FileNotFoundError:
    st.warning(f"Image not found locally or on GitHub. Please check the image path or internet connection.")
except Exception as e:
    st.error(f"Error loading image: {str(e)}")

# Function to recreate the database
def recreate_database() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Create the database and required tables if they do not exist.
    
    Returns:
        tuple: (connection, cursor)
    """
    conn = sqlite3.connect("portfolio1.1.db")
    c = conn.cursor()
    
    # Create separate tables for each investment type if they do not already exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker TEXT,
            AmountInvested REAL,
            PurchasePrice REAL,
            CurrentPrice REAL,
            CurrentValue REAL,
            StartDate TEXT,
            EndDate TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker TEXT,
            AmountInvested REAL,
            PurchasePrice REAL,
            CurrentPrice REAL,
            CurrentValue REAL,
            StartDate TEXT,
            EndDate TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS private_placements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Company TEXT,
            Industry TEXT,
            Revenue REAL,
            Valuation REAL,
            Ownership REAL,
            AmountInvested REAL,
            DateInvested TEXT,
            Platform TEXT,
            Repayment TEXT,
            Frequency TEXT,
            TotalReturn REAL,
            ExitDate TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS startups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Company TEXT,
            Industry TEXT,
            Founders TEXT,
            Website TEXT,
            Email TEXT,
            FundRaise REAL,
            PitchSummary TEXT,
            Views INTEGER,
            Valuation REAL,
            AmountInvested REAL,
            Ownership REAL,
            DateInvested TEXT,
            Method TEXT,
            ExitDate TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS brick_and_mortar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Company TEXT,
            Industry TEXT,
            Revenue REAL,
            Valuation REAL,
            Investors TEXT,
            IbrahimAliyu REAL,
            FarukAliyu REAL,
            AkhCapital REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fx (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Currency TEXT,
            ExchangeRate REAL,
            EntryPrice REAL,
            EntryDate TEXT,
            TargetExit REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS mudarabah (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Company TEXT,
            Industry TEXT,
            FundRaise REAL,
            Investors TEXT,
            AmountInvested REAL,
            AkhOwnership REAL,
            DateInvested TEXT,
            Repayment TEXT,
            Frequency TEXT,
            TotalReturn REAL,
            ExitDate TEXT
        )
    """)
    conn.commit()
    
    # Schema migration: add missing columns to stocks if not already present
    try:
        c.execute("ALTER TABLE stocks ADD COLUMN PurchasePrice REAL")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    try:
        c.execute("ALTER TABLE stocks ADD COLUMN CurrentPrice REAL")
    except sqlite3.OperationalError:
        pass

    # Schema migration: add missing columns to crypto if not already present
    try:
        c.execute("ALTER TABLE crypto ADD COLUMN PurchasePrice REAL")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE crypto ADD COLUMN CurrentPrice REAL")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn, c

# Function to fetch stock data
def fetch_stock_data(ticker: str, start_date: Any, end_date: Any) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker.
        start_date (Any): Start date.
        end_date (Any): End date.
    
    Returns:
        pd.DataFrame: DataFrame with stock history.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"Invalid or delisted stock ticker: {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to fetch crypto data
def fetch_crypto_data(ticker: str, start_date: Any, end_date: Any) -> pd.DataFrame:
    """
    Fetch crypto data from Yahoo Finance.
    
    Args:
        ticker (str): Crypto ticker.
        start_date (Any): Start date.
        end_date (Any): End date.
    
    Returns:
        pd.DataFrame: DataFrame with crypto history.
    """
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"Invalid or delisted crypto ticker: {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching crypto data for {ticker}: {e}")
        st.error(f"Error fetching crypto data: {e}")
        return pd.DataFrame()

# Function to display data from a specific table
def display_table_data(table_name: str) -> None:
    """
    Display data from a specific table.
    
    Args:
        table_name (str): Name of the table.
    """
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
            st.dataframe(df)
        else:
            st.info(f"No data available in the {table_name} table.")
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve data from {table_name}: {e}")
        st.error(f"Failed to retrieve data from {table_name}: {e}")

# Function to generate interactive visualizations using Plotly
def generate_plotly_line_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Generate a Plotly line chart.
    
    Args:
        data (pd.DataFrame): DataFrame with data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Chart title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    if data.empty:
        st.warning(f"No data available for {title}")
        return
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: xlabel, y_col: ylabel},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_plotly_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Generate a Plotly bar chart.
    
    Args:
        data (pd.DataFrame): DataFrame with data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Chart title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    if data.empty:
        st.warning(f"No data available for {title}")
        return
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: xlabel, y_col: ylabel},
        text=y_col
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# Modified Function: generate_plotly_pie_chart to create a breakout donut pie chart
def generate_plotly_pie_chart(data: pd.DataFrame, names_col: str, values_col: str, title: str) -> None:
    """
    Generate a Plotly pie chart with a breakout donut effect.
    
    Args:
        data (pd.DataFrame): DataFrame with data.
        names_col (str): Column name for pie chart labels.
        values_col (str): Column name for pie chart values.
        title (str): Chart title.
    """
    if data.empty:
        st.warning(f"No data available for {title}")
        return
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4  # donut effect
    )
    fig.update_traces(textinfo='percent+label', pull=0.1)  # breakout effect
    st.plotly_chart(fig, use_container_width=True)

# Function to generate a line chart with a trend line
def generate_plotly_line_chart_with_trend(data: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Generate a Plotly line chart with a trend line.
    
    Args:
        data (pd.DataFrame): DataFrame with data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Chart title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    if data.empty:
        st.warning(f"No data available for {title}")
        return
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: xlabel, y_col: ylabel},
        trendline="ols",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig, use_container_width=True)

# Add context manager for database connection
@st.cache_resource
def init_connection() -> sqlite3.Connection:
    """
    Initialize the database connection.
    
    Returns:
        sqlite3.Connection: Database connection.
    """
    return sqlite3.connect("portfolio1.1.db", check_same_thread=False)

# Database connection with error handling
try:
    conn = init_connection()
    cursor = conn.cursor()
    recreate_database()
except sqlite3.Error as e:
    logger.error(f"Database connection failed: {e}")
    st.error(f"Database connection failed: {e}")
    st.stop()

# Sidebar Navigation
investment_type = st.sidebar.selectbox("Select Investment Type", [
    "Stocks", "Crypto", "Private Placements", "Startups", "Brick & Mortar", "FX", "Mudarabah"
])

# Add a reset button with confirmation
if st.sidebar.button("Reset All Tables"):
    if st.sidebar.checkbox("Confirm Reset"):
        try:
            for table in ["stocks", "crypto", "private_placements", "startups", "brick_and_mortar", "fx", "mudarabah"]:
                cursor.execute(f"DELETE FROM {table}")
            conn.commit()
            st.success("All tables have been reset.")
        except sqlite3.Error as e:
            logger.error(f"Failed to reset tables: {e}")
            st.error(f"Failed to reset tables: {e}")

# Remove Portfolio Overview section and add visualization function for investment summaries
def display_investment_summary(investment_type: str, query: str, value_column: str = "AmountInvested") -> None:
    """
    Display investment summary with visualizations.
    
    Args:
        investment_type (str): Type of investment.
        query (str): SQL query to fetch data.
        value_column (str): Column name for values. Default is "AmountInvested".
    """
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Category", "Value"])
            total_value = df["Value"].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Total {investment_type} Value", f"${total_value:,.2f}")
                generate_plotly_pie_chart(
                    df,
                    names_col="Category",
                    values_col="Value",
                    title=f"{investment_type} Allocation"
                )
            with col2:
                generate_plotly_bar_chart(
                    df,
                    x_col="Category",
                    y_col="Value",
                    title=f"{investment_type} Distribution",
                    xlabel="Category",
                    ylabel="Value ($)"
                )
    except Exception as e:
        logger.error(f"Error displaying {investment_type} summary: {e}")
        st.error(f"Error displaying {investment_type} summary: {e}")

# Function to display a modal form for adding new entries
def display_add_form(investment_type: str, form_fields: list, insert_query: str, success_message: str) -> None:
    """
    Display a modal form for adding new entries.
    
    Args:
        investment_type (str): Type of investment.
        form_fields (list): List of form fields.
        insert_query (str): SQL query to insert data.
        success_message (str): Success message.
    """
    with st.expander(f"Add New {investment_type}", expanded=True):
        with st.form(key=f"{investment_type}_form"):
            inputs = {}
            for field_name, field_type, field_args in form_fields:
                if field_type == "text":
                    inputs[field_name] = st.text_input(field_args["label"])
                elif field_type == "number":
                    inputs[field_name] = st.number_input(field_args["label"], **field_args.get("kwargs", {}))
                elif field_type == "date":
                    inputs[field_name] = st.date_input(field_args["label"])
            
            submit_button = st.form_submit_button(label="Add")
            if submit_button:
                try:
                    # AUTOMATED PURCHASE PRICE: for stock/crypto, fetch price from start_date to start_date + 1 day.
                    if investment_type.lower() in ["stock", "crypto"]:
                        ticker = inputs["Ticker"].strip().upper()
                        start_date_obj = inputs["StartDate"]
                        end_date_obj = inputs["EndDate"]
                        start_date_str = start_date_obj.strftime("%Y-%m-%d")
                        end_date_str = end_date_obj.strftime("%Y-%m-%d")
                        # Fetch purchase price using start date
                        next_day_str = (start_date_obj + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        if investment_type.lower() == "stock":
                            purchase_data = fetch_stock_data(ticker, start_date_str, next_day_str)
                        else:
                            purchase_data = fetch_crypto_data(ticker, start_date_str, next_day_str)
                        purchase_price = purchase_data["Close"].iloc[0] if not purchase_data.empty else 0
                        # Fetch current price using end date
                        following_day_str = (end_date_obj + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        if investment_type.lower() == "stock":
                            current_data = fetch_stock_data(ticker, end_date_str, following_day_str)
                        else:
                            current_data = fetch_crypto_data(ticker, end_date_str, following_day_str)
                        current_price = current_data["Close"].iloc[-1] if not current_data.empty else 0
                        current_value = (inputs["AmountInvested"] * current_price / purchase_price) if purchase_price != 0 else 0
                        new_values = (
                            inputs["Ticker"],
                            inputs["AmountInvested"],
                            purchase_price,
                            current_price,
                            current_value,
                            start_date_str,
                            end_date_str
                        )
                        cursor.execute(insert_query, new_values)
                        st.info(f"Purchase Price on {start_date_str}: ${purchase_price:.2f}")
                        st.info(f"Current Price on {end_date_str}: ${current_price:.2f}")
                    else:
                        cursor.execute(insert_query, tuple(inputs.values()))
                    conn.commit()
                    st.success(success_message)
                    
                    # Update current value for stocks and crypto immediately after insertion
                    if investment_type.lower() == "stock":
                        # Convert dates to string format for yfinance
                        start_date = inputs["StartDate"].strftime("%Y-%m-%d")
                        end_date = min(
                            inputs["EndDate"],
                            pd.Timestamp.today().date()
                        ).strftime("%Y-%m-%d")
                        
                        data = fetch_stock_data(
                            inputs["Ticker"],
                            start_date,
                            end_date
                        )
                        if not data.empty:
                            current_value = data["Close"].iloc[-1]
                            cursor.execute("""
                                UPDATE stocks 
                                SET CurrentValue = ? 
                                WHERE id = (SELECT MAX(id) FROM stocks)
                            """, (current_value,))
                            conn.commit()
                            st.info(f"Updated current value to: ${current_value:,.2f}")
                            
                    elif investment_type.lower() == "crypto":
                        # Convert dates to string format for yfinance
                        start_date = inputs["StartDate"].strftime("%Y-%m-%d")
                        end_date = min(
                            inputs["EndDate"],
                            pd.Timestamp.today().date()
                        ).strftime("%Y-%m-%d")
                        
                        data = fetch_crypto_data(
                            inputs["Ticker"],
                            start_date,
                            end_date
                        )
                        if not data.empty:
                            current_price = data["Close"].iloc[-1]
                            # Calculate current value based on invested amount and current price
                            initial_price = data["Close"].iloc[0]
                            crypto_amount = inputs["AmountInvested"] / initial_price
                            current_value = crypto_amount * current_price
                            
                            cursor.execute("""
                                UPDATE crypto 
                                SET CurrentValue = ? 
                                WHERE id = (SELECT MAX(id) FROM crypto)
                            """, (current_value,))
                            conn.commit()
                            st.info(f"Updated current value to: ${current_value:,.2f}")
                            if current_value > inputs["AmountInvested"]:
                                st.success(f"Profit: ${current_value - inputs['AmountInvested']:,.2f} (+{((current_value/inputs['AmountInvested'])-1)*100:.1f}%)")
                            else:
                                st.error(f"Loss: ${current_value - inputs['AmountInvested']:,.2f} ({((current_value/inputs['AmountInvested'])-1)*100:.1f}%)")
                            
                except sqlite3.Error as e:
                    logger.error(f"Failed to add {investment_type} entry: {e}")
                    st.error(f"Failed to add {investment_type} entry: {e}")

# NEW: Function to delete an investment entry
def delete_investment(table: str, entry_id: int) -> None:
    """
    Delete an investment entry.
    
    Args:
        table (str): Table name.
        entry_id (int): Entry ID.
    """
    try:
        cursor.execute(f"DELETE FROM {table} WHERE id = ?", (entry_id,))
        conn.commit()
        st.success(f"Investment ID {entry_id} from {table} has been closed and deleted successfully!")
    except sqlite3.Error as e:
        logger.error(f"Failed to delete investment {entry_id} from {table}: {e}")
        st.error(f"Failed to delete investment {entry_id} from {table}: {e}")

# NEW: Section to mark an investment as closed and delete it
with st.expander("Close Investment Position"):
    st.info("Mark an investment as closed and remove it from the portfolio")
    investment_option = st.selectbox("Select Investment Type", 
        ["stocks", "crypto", "private_placements", "startups", "brick_and_mortar", "fx", "mudarabah"])
    entry_id = st.number_input("Enter Investment ID to Close", min_value=1, step=1)
    if st.button("Close Investment"):
        delete_investment(investment_option, int(entry_id))

# Add new function for editable investment tables
def display_editable_table(table_name: str) -> None:
    """Display an editable table and handle updates."""
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        if not data:
            st.info(f"No data available in {table_name} table.")
            return
            
        # Create DataFrame with column names
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        
        # Create editable table
        edited_df = st.data_editor(df, num_rows="dynamic", key=f"editor_{table_name}")
        
        # Add save button
        if st.button(f"Save Changes to {table_name}"):
            for index, row in edited_df.iterrows():
                # Get original row for comparison
                original_row = df.iloc[index] if index < len(df) else None
                
                if original_row is not None and not row.equals(original_row):
                    # Update existing row
                    set_clause = ", ".join([f"{col} = ?" for col in columns[1:]])  # Skip ID column
                    values = [row[col] for col in columns[1:]]  # Values without ID
                    values.append(row['id'])  # Add ID for WHERE clause
                    
                    query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"
                    cursor.execute(query, values)
            
            conn.commit()
            st.success(f"Changes saved to {table_name} successfully!")
    except Exception as e:
        st.error(f"Error updating {table_name}: {e}")

# Stocks Section
if investment_type == "Stocks":
    # Add button and form for adding new stock entries
    display_add_form(
        "Stock",
        form_fields=[
            ("Ticker", "text", {"label": "Enter Stock Ticker (e.g., AAPL, TSLA)"}),
            ("AmountInvested", "number", {"label": "Investment Amount ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            # Removed PurchasePrice field for automation
            ("StartDate", "date", {"label": "Start Date"}),
            ("EndDate", "date", {"label": "End Date"})
        ],
        insert_query="""
            INSERT INTO stocks (Ticker, AmountInvested, PurchasePrice, CurrentPrice, CurrentValue, StartDate, EndDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Stock investment added successfully!"
    )
    
    display_table_data("stocks")
    
    # Portfolio Group Summary
    st.subheader("Stock Portfolio Overview")
    cursor.execute("""
        SELECT Ticker, SUM(AmountInvested) as TotalInvested, SUM(CurrentValue) as TotalValue
        FROM stocks 
        GROUP BY Ticker
        HAVING TotalInvested > 0
    """)
    portfolio_data = cursor.fetchall()
    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data, columns=["Ticker", "TotalInvested", "TotalValue"])
        col1, col2 = st.columns(2)
        with col1:
            generate_plotly_pie_chart(
                portfolio_df,
                names_col="Ticker",
                values_col="TotalValue",
                title="Portfolio Distribution"
            )
        with col2:
            st.metric(
                "Total Portfolio Value", 
                f"${portfolio_df['TotalValue'].sum():,.2f}",
                f"{((portfolio_df['TotalValue'].sum() / portfolio_df['TotalInvested'].sum()) - 1) * 100:.1f}%"
            )
    
    # Updated Price Trend Analysis for All Stocks (Normalized Gains)
    # Automatically set start_date as the earliest StartDate among stock records
    cursor.execute("SELECT MIN(StartDate) FROM stocks")
    min_date_str = cursor.fetchone()[0]
    if min_date_str:
        start_date_default = pd.to_datetime(min_date_str)
    else:
        start_date_default = pd.to_datetime("today")
    end_date_default = pd.to_datetime("today")
    
    st.subheader("Price Trend Analysis for All Stocks (Normalized Gains)")
    start_date = st.date_input("Start Date (Auto)", value=start_date_default, key="stock_all_start", disabled=True)
    end_date = st.date_input("End Date (Auto)", value=end_date_default, key="stock_all_end", disabled=True)
    
    cursor.execute("SELECT DISTINCT Ticker FROM stocks")
    tickers = [row[0] for row in cursor.fetchall()]
    all_stock_data = pd.DataFrame()
    if tickers:
        for ticker in tickers:
            data = fetch_stock_data(ticker, start_date, end_date)
            if not data.empty:
                data = data.reset_index()
                data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")  # Convert Date to string
                data["Ticker"] = ticker
                # Compute normalized gain: ((current price / initial price) - 1) * 100
                data["BasePrice"] = data.groupby("Ticker")["Close"].transform("first")
                data["Gain"] = (data["Close"] / data["BasePrice"] - 1) * 100
                all_stock_data = pd.concat([all_stock_data, data], ignore_index=True)
    if not all_stock_data.empty:
        fig = px.line(
            all_stock_data,
            x="Date",
            y="Gain",
            color="Ticker",
            title="Stock Normalized Gains for All Tickers",
            labels={"Date": "Date", "Gain": "Gain (%)", "Ticker": "Ticker"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Crypto Section
if investment_type == "Crypto":
    # Add button and form for adding new crypto entries
    display_add_form(
        "Crypto",
        form_fields=[
            ("Ticker", "text", {"label": "Enter Crypto Ticker (e.g., BTC-USD, ETH-USD)"}),
            ("AmountInvested", "number", {"label": "Investment Amount ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            # Removed PurchasePrice field for automation
            ("StartDate", "date", {"label": "Start Date"}),
            ("EndDate", "date", {"label": "End Date"})
        ],
        insert_query="""
            INSERT INTO crypto (Ticker, AmountInvested, PurchasePrice, CurrentPrice, CurrentValue, StartDate, EndDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Crypto investment added successfully!"
    )
    
    display_table_data("crypto")
    
    # Portfolio Group Summary
    st.subheader("Crypto Portfolio Overview")
    cursor.execute("""
        SELECT Ticker, SUM(AmountInvested) as TotalInvested, SUM(CurrentValue) as TotalValue
        FROM crypto 
        GROUP BY Ticker
        HAVING TotalInvested > 0
    """)
    portfolio_data = cursor.fetchall()
    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data, columns=["Ticker", "TotalInvested", "TotalValue"])
        col1, col2 = st.columns(2)
        with col1:
            generate_plotly_pie_chart(
                portfolio_df,
                names_col="Ticker",
                values_col="TotalValue",
                title="Portfolio Distribution"
            )
        with col2:
            st.metric(
                "Total Portfolio Value", 
                f"${portfolio_df['TotalValue'].sum():,.2f}",
                f"{((portfolio_df['TotalValue'].sum() / portfolio_df['TotalInvested'].sum()) - 1) * 100:.1f}%"
            )
    
    # Updated Price Trend Analysis for All Cryptos (Normalized Gains)
    # Automatically set start_date as the earliest StartDate among crypto records
    cursor.execute("SELECT MIN(StartDate) FROM crypto")
    min_date_str = cursor.fetchone()[0]
    if min_date_str:
        start_date_default = pd.to_datetime(min_date_str)
    else:
        start_date_default = pd.to_datetime("today")
    end_date_default = pd.to_datetime("today")
    
    st.subheader("Price Trend Analysis for All Cryptos (Normalized Gains)")
    start_date = st.date_input("Start Date (Auto)", value=start_date_default, key="crypto_all_start", disabled=True)
    end_date = st.date_input("End Date (Auto)", value=end_date_default, key="crypto_all_end", disabled=True)
    
    cursor.execute("SELECT DISTINCT Ticker FROM crypto")
    tickers = [row[0] for row in cursor.fetchall()]
    all_crypto_data = pd.DataFrame()
    if tickers:
        for ticker in tickers:
            data = fetch_crypto_data(ticker, start_date, end_date)
            if not data.empty:
                data = data.reset_index()
                data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")  # convert Date to string
                data["Ticker"] = ticker
                # Compute normalized gain: ((current price / initial price) - 1) * 100
                data["BasePrice"] = data.groupby("Ticker")["Close"].transform("first")
                data["Gain"] = (data["Close"] / data["BasePrice"] - 1) * 100
                all_crypto_data = pd.concat([all_crypto_data, data], ignore_index=True)
    if not all_crypto_data.empty:
        fig = px.line(
            all_crypto_data,
            x="Date",
            y="Gain",
            color="Ticker",
            title="Crypto Normalized Gains for All Tickers",
            labels={"Date": "Date", "Gain": "Gain (%)", "Ticker": "Ticker"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Private Placements Section
if investment_type == "Private Placements":
    # Add button and form for adding new private placement entries
    display_add_form(
        "Private Placement",
        form_fields=[
            ("Company", "text", {"label": "Company"}),
            ("Industry", "text", {"label": "Industry"}),
            ("Revenue", "number", {"label": "Revenue ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Valuation", "number", {"label": "Valuation ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Ownership", "number", {"label": "Ownership (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}}),
            ("AmountInvested", "number", {"label": "Investment Amount ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("DateInvested", "date", {"label": "Date Invested"}),
            ("Platform", "text", {"label": "Platform"}),
            ("Repayment", "text", {"label": "Repayment"}),
            ("Frequency", "text", {"label": "Frequency"}),
            ("TotalReturn", "number", {"label": "Total Return ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("ExitDate", "date", {"label": "Exit Date"})
        ],
        insert_query="""
            INSERT INTO private_placements 
            (Company, Industry, Revenue, Valuation, Ownership, AmountInvested, DateInvested, Platform, Repayment, Frequency, TotalReturn, ExitDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Private placement added successfully!"
    )
    
    display_editable_table("private_placements")
    
    # Generate Plotly visualization for private placements
    cursor.execute("SELECT Company, Valuation FROM private_placements")
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Company", "Valuation"])
        generate_plotly_bar_chart(
            df,
            x_col="Company",
            y_col="Valuation",
            title="Private Placements Valuation",
            xlabel="Company",
            ylabel="Valuation ($)"
        )
    
    # Update private placements summary query
    pp_query = """
        SELECT Company as Category, 
               COALESCE(Valuation * (Ownership / 100.0), 0) as Value 
        FROM private_placements
    """
    display_investment_summary("Private Placements", pp_query)

# Startups Section
if investment_type == "Startups":
    # Add button and form for adding new startup entries
    display_add_form(
        "Startup",
        form_fields=[
            ("Company", "text", {"label": "Company"}),
            ("Industry", "text", {"label": "Industry (i.e. subsector)"}),
            ("Founders", "text", {"label": "Founder(s)"}),
            ("Website", "text", {"label": "Website"}),
            ("Email", "text", {"label": "Email"}),
            ("FundRaise", "number", {"label": "Fund Raise ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("PitchSummary", "text", {"label": "Pitch Summary"}),
            ("Views", "number", {"label": "Views", "kwargs": {"min_value": 0}}),
            ("Valuation", "number", {"label": "Valuation ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("AmountInvested", "number", {"label": "Amount Invested ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Ownership", "number", {"label": "Ownership (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}}),
            ("DateInvested", "date", {"label": "Date Invested"}),
            ("Method", "text", {"label": "Method"}),
            ("ExitDate", "date", {"label": "Exit Date"})
        ],
        insert_query="""
            INSERT INTO startups (Company, Industry, Founders, Website, Email, FundRaise, PitchSummary, Views, Valuation, AmountInvested, Ownership, DateInvested, Method, ExitDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Startup investment added successfully!"
    )
    
    display_editable_table("startups")
    
    # Generate Plotly visualization for startups
    cursor.execute("SELECT Company, FundRaise FROM startups")
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Company", "FundRaise"])
        generate_plotly_bar_chart(
            df,
            x_col="Company",
            y_col="FundRaise",
            title="Startups Fund Raise",
            xlabel="Company",
            ylabel="Fund Raise ($)"
        )
    
    # Add startups summary
    startups_query = """
        SELECT Company as Category, AmountInvested as Value 
        FROM startups
    """
    display_investment_summary("Startups", startups_query)

# Brick & Mortar Section
if investment_type == "Brick & Mortar":
    # Add button and form for adding new brick & mortar entries
    display_add_form(
        "Brick & Mortar",
        form_fields=[
            ("Company", "text", {"label": "Company"}),
            ("Industry", "text", {"label": "Industry (i.e. subsector)"}),
            ("Revenue", "number", {"label": "Revenue (MRQ) ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Valuation", "number", {"label": "Valuation ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Investors", "text", {"label": "Investors"}),
            ("IbrahimAliyu", "number", {"label": "Ibrahim Aliyu (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}}),
            ("FarukAliyu", "number", {"label": "Faruk Aliyu (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}}),
            ("AkhCapital", "number", {"label": "Akh Capital (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}})
        ],
        insert_query="""
            INSERT INTO brick_and_mortar 
            (Company, Industry, Revenue, Valuation, Investors, IbrahimAliyu, FarukAliyu, AkhCapital) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Brick & Mortar investment added successfully!"
    )
    
    display_editable_table("brick_and_mortar")
    
    # Generate Plotly visualization for brick and mortar
    cursor.execute("SELECT Company, Revenue FROM brick_and_mortar")
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Company", "Revenue"])
        generate_plotly_bar_chart(
            df,
            x_col="Company",
            y_col="Revenue",
            title="Brick & Mortar Revenue",
            xlabel="Company",
            ylabel="Revenue ($)"
        )
    
    # Update brick & mortar summary query
    bm_query = """
        SELECT Company as Category, 
               COALESCE(Revenue, 0) as Value 
        FROM brick_and_mortar
    """
    display_investment_summary("Brick & Mortar", bm_query)

# FX Section
if investment_type == "FX":
    # Add button and form for adding new FX entries
    display_add_form(
        "FX",
        form_fields=[
            ("Currency", "text", {"label": "Currency"}),
            ("ExchangeRate", "number", {"label": "Exchange Rate (current)", "kwargs": {"min_value": 0.0, "step": 0.01}}),
            ("EntryPrice", "number", {"label": "Entry Price", "kwargs": {"min_value": 0.0, "step": 0.01}}),
            ("EntryDate", "date", {"label": "Entry Date"}),
            ("TargetExit", "number", {"label": "Target Exit", "kwargs": {"min_value": 0.0, "step": 0.01}})
        ],
        insert_query="""
            INSERT INTO fx 
            (Currency, ExchangeRate, EntryPrice, EntryDate, TargetExit) 
            VALUES (?, ?, ?, ?, ?)
        """,
        success_message="FX investment added successfully!"
    )
    
    display_editable_table("fx")
    
    # Generate Plotly visualization for FX
    cursor.execute("SELECT Currency, ExchangeRate FROM fx")
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Currency", "ExchangeRate"])
        generate_plotly_bar_chart(
            df,
            x_col="Currency",
            y_col="ExchangeRate",
            title="FX Exchange Rates",
            xlabel="Currency",
            ylabel="Exchange Rate"
        )
    
    # Update FX summary query
    fx_query = """
        SELECT Currency as Category, 
               COALESCE(ExchangeRate, 0) as Value 
        FROM fx
    """
    display_investment_summary("FX", fx_query)

# Mudarabah Section
if investment_type == "Mudarabah":
    # Add button and form for adding new mudarabah entries
    display_add_form(
        "Mudarabah",
        form_fields=[
            ("Company", "text", {"label": "Company"}),
            ("Industry", "text", {"label": "Industry (i.e. subsector)"}),
            ("FundRaise", "number", {"label": "Fund Raise ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("Investors", "text", {"label": "Investors"}),
            ("AmountInvested", "number", {"label": "Amount Invested ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("AkhOwnership", "number", {"label": "Akh Ownership (%)", "kwargs": {"min_value": 0.0, "max_value": 100.0, "step": 0.1}}),
            ("DateInvested", "date", {"label": "Date Invested"}),
            ("Repayment", "text", {"label": "Repayment"}),
            ("Frequency", "text", {"label": "Frequency"}),
            ("TotalReturn", "number", {"label": "Total Return ($)", "kwargs": {"min_value": 0.0, "step": 100.0}}),
            ("ExitDate", "date", {"label": "Exit Date"})
        ],
        insert_query="""
            INSERT INTO mudarabah (Company, Industry, FundRaise, Investors, AmountInvested, AkhOwnership, DateInvested, Repayment, Frequency, TotalReturn, ExitDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        success_message="Mudarabah investment added successfully!"
    )
    
    display_editable_table("mudarabah")
    
    # Generate Plotly visualization for mudarabah
    cursor.execute("SELECT Company, TotalReturn FROM mudarabah")
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Company", "TotalReturn"])
        generate_plotly_bar_chart(
            df,
            x_col="Company",
            y_col="TotalReturn",
            title="Mudarabah Total Returns",
            xlabel="Company",
            ylabel="Total Return ($)"
        )
    
    # Add mudarabah summary
    mudarabah_query = """
        SELECT Company as Category, AmountInvested as Value 
        FROM mudarabah
    """
    display_investment_summary("Mudarabah", mudarabah_query)

# Add custom CSS for UI polish (optional)
st.markdown(
    """
    <style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stButton>button {background-color: #1f77b4; color: #fff; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

def update_current_value_stock(inputs: dict) -> None:
    ticker = inputs["Ticker"].strip().upper()  # ensure ticker is uppercase
    start_date = pd.to_datetime(inputs["StartDate"])
    end_date = pd.to_datetime(inputs["EndDate"])
    today = pd.to_datetime("today")
    if end_date > today:
        end_date = today
        st.info("End date is in the future. Using today's date for current value calculation.")
    st.write(f"[DEBUG] Fetching stock data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.write("[DEBUG] Fetched Data (last 5 rows):", data.tail())
    if not data.empty:
        current_price = data["Close"].iloc[-1]
        st.write(f"[DEBUG] Calculated current price for {ticker}: {current_price}")
        cursor.execute(
            "UPDATE stocks SET CurrentPrice = ? WHERE UPPER(Ticker) = ? AND id = (SELECT MAX(id) FROM stocks WHERE UPPER(Ticker) = ?)",
            (current_price, ticker, ticker)
        )
        conn.commit()
        st.success(f"Updated current price for {ticker}: ${current_price:.2f}")
    else:
        st.warning(f"Unable to fetch current price for {ticker}")

def update_current_value_crypto(inputs: dict) -> None:
    ticker = inputs["Ticker"].strip().upper()  # ensure ticker is uppercase
    start_date = pd.to_datetime(inputs["StartDate"])
    end_date = pd.to_datetime(inputs["EndDate"])
    today = pd.to_datetime("today")
    if end_date > today:
        end_date = today
        st.info("End date is in the future. Using today's date for current value calculation.")
    st.write(f"[DEBUG] Fetching crypto data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = fetch_crypto_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.write("[DEBUG] Fetched Data (last 5 rows):", data.tail())
    if not data.empty:
        current_price = data["Close"].iloc[-1]
        initial_price = data["Close"].iloc[0]
        # Show purchase price for crypto
        st.info(f"Purchase Price on {start_date.strftime('%Y-%m-%d')}: ${initial_price:.2f}")
        # Get the invested amount from the database
        cursor.execute(
            "SELECT AmountInvested FROM crypto WHERE UPPER(Ticker) = ? AND id = (SELECT MAX(id) FROM crypto WHERE UPPER(Ticker) = ?)",
            (ticker, ticker)
        )
        amount_invested = cursor.fetchone()[0]
        # Calculate crypto amount and current value
        crypto_amount = amount_invested / initial_price
        current_value = crypto_amount * current_price
        cursor.execute(
            "UPDATE crypto SET CurrentPrice = ? WHERE UPPER(Ticker) = ? AND id = (SELECT MAX(id) FROM crypto WHERE UPPER(Ticker) = ?)",
            (current_price, ticker, ticker)
        )
        conn.commit()
        st.success(f"Updated current value for {ticker}: ${current_value:.2f}")
        # Show profit/loss
        if current_value > amount_invested:
            st.success(f"Profit: ${current_value - amount_invested:,.2f} (+{((current_value/amount_invested)-1)*100:.1f}%)")
        else:
            st.error(f"Loss: ${current_value - amount_invested:,.2f} ({((current_value/amount_invested)-1)*100:.1f}%)")
    else:
        st.warning(f"Unable to fetch current price for {ticker}")

# Test section to simulate current value updates for stock and crypto
if st.sidebar.checkbox("Run Update Tests"):
    st.subheader("Testing Current Value Updates")
    # Test data for stock (adjust ticker and dates as needed)
    test_stock = {"Ticker": "AAPL", "StartDate": "2021-01-04", "EndDate": "2025-03-26"}
    st.write("Testing Stock Update:", test_stock)
    update_current_value_stock(test_stock)
    
    # Test data for crypto (adjust ticker and dates as needed)
    test_crypto = {"Ticker": "BTC-USD", "StartDate": "2021-01-04", "EndDate": "2025-03-26"}
    st.write("Testing Crypto Update:", test_crypto)
    update_current_value_crypto(test_crypto)