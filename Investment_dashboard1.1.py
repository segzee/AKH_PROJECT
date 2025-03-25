import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import sqlite3
import os

st.set_page_config(page_title="Investment Dashboard", layout="wide")
st.title("📈 Multi-Asset Investment Dashboard")

# Function to recreate the database
def recreate_database():
    if os.path.exists("portfolio1.1.db"):
        try:
            conn = sqlite3.connect("portfolio1.1.db")
            conn.close()  # Close the connection if it exists
            os.rename("portfolio1.1.db", "portfolio1.1_backup.db")  # Rename instead of delete
        except sqlite3.Error as e:
            st.error(f"Failed to close the database connection: {e}")
        except OSError as e:
            st.error(f"Failed to rename the database file: {e}")
    conn = sqlite3.connect("portfolio1.1.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Type TEXT,
            Ticker TEXT,
            AmountInvested REAL,
            CurrentValue REAL,
            AdditionalInfo TEXT  -- New column for additional information
        )
    """)
    conn.commit()
    return conn, c

# Function to verify the database schema
def verify_database_schema(c):
    try:
        c.execute("PRAGMA table_info(portfolio)")
        columns = [col[1] for col in c.fetchall()]
        expected_columns = ["id", "Type", "Ticker", "AmountInvested", "CurrentValue", "AdditionalInfo"]
        if set(columns) != set(expected_columns):  # Using set comparison for more flexibility
            raise sqlite3.Error("Database schema does not match expected schema.")
    except sqlite3.Error as e:
        st.error(f"Database schema verification failed: {e}")
        return False
    return True

# Database connection with error handling
try:
    conn = sqlite3.connect("portfolio1.1.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'")
    if not c.fetchone() or not verify_database_schema(c):
        conn, c = recreate_database()
except sqlite3.Error as e:
    st.error(f"Database connection failed: {e}")
    conn, c = recreate_database()

# Sidebar Navigation
investment_type = st.sidebar.selectbox("Select Investment Type", [
    "Portfolio Overview", "Stocks", "Crypto", "Startups", "Private Placements", "Brick & Mortar", "FX", "Mudarabah"
])

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"Invalid or delisted stock ticker: {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to fetch crypto data
def fetch_crypto_data(ticker, start_date, end_date):
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"Invalid or delisted crypto ticker: {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return pd.DataFrame()

# Add a reset button with confirmation
if st.sidebar.button("Reset Portfolio"):
    if st.sidebar.checkbox("Confirm Reset"):
        try:
            c.execute("DELETE FROM portfolio")
            conn.commit()
            st.success("Portfolio has been reset.")
        except sqlite3.Error as e:
            st.error(f"Failed to reset portfolio: {e}")

# Function to display portfolio overview for a specific investment type
def display_portfolio_overview(investment_type=None):
    try:
        if investment_type:
            c.execute("SELECT * FROM portfolio WHERE Type = ?", (investment_type,))
        else:
            c.execute("SELECT * FROM portfolio")
        data = c.fetchall()
        
        if data:
            df = pd.DataFrame(data, columns=["ID", "Type", "Ticker", "Amount Invested", "Current Value", "Additional Info"])
            df["Return ($)"] = df["Current Value"] - df["Amount Invested"]
            df["Return (%)"] = (df["Return ($)"] / df["Amount Invested"]) * 100
            
            st.subheader(f"{investment_type} Portfolio" if investment_type else "Your Investment Portfolio")
            st.dataframe(df.drop(columns=["ID"]))
            
            total_invested = df["Amount Invested"].sum()
            total_value = df["Current Value"].sum()
            total_return = total_value - total_invested
            total_return_percent = (total_return / total_invested) * 100 if total_invested > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Portfolio Value ($)", value=f"{total_value:,.2f}")
            with col2:
                st.metric(label="Total Return ($)", value=f"{total_return:,.2f}")
            with col3:
                st.metric(label="Total Return (%)", value=f"{total_return_percent:.2f}%")
            
            # Add portfolio allocation chart
            st.subheader("Portfolio Allocation")
            fig = px.pie(df, names="Ticker", values="Current Value", title=f"{investment_type} Allocation" if investment_type else "Portfolio Allocation by Investment Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Add returns by investment type
            st.subheader("Returns by Investment Type")
            fig2 = px.bar(df, 
                         x="Ticker", y="Return ($)", 
                         color="Ticker",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig2, use_container_width=True)

            # Add delete option for each entry
            for index, row in df.iterrows():
                if st.button(f"Delete {row['Type']} - {row['Ticker']}", key=f"delete_{row['ID']}"):
                    try:
                        c.execute("DELETE FROM portfolio WHERE id = ?", (row["ID"],))
                        conn.commit()
                        st.success(f"Deleted {row['Type']} - {row['Ticker']} from portfolio.")
                    except sqlite3.Error as e:
                        st.error(f"Failed to delete {row['Type']} - {row['Ticker']}: {e}")
        else:
            st.info("Your portfolio is empty. Please add investments to see your portfolio overview.")
    except sqlite3.Error as e:
        st.error(f"Failed to retrieve portfolio data: {e}")

# Stocks Section
if investment_type == "Stocks":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, step=100.0)
    
    if st.sidebar.button("Add Stock Investment"):
        if ticker and investment_amount > 0:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                current_price = stock_data["Close"].iloc[-1]
                try:
                    c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue) VALUES (?, ?, ?, ?)",
                              ("Stock", ticker, investment_amount, investment_amount * (current_price / stock_data["Close"].iloc[0])))
                    conn.commit()
                    st.sidebar.success(f"Added {ticker} investment successfully!")
                except sqlite3.Error as e:
                    st.error(f"Failed to add stock investment: {e}")
    
    if ticker:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if not stock_data.empty:
            st.subheader(f"{ticker} Stock Price")
            st.line_chart(stock_data["Close"], use_container_width=True)
    
    display_portfolio_overview("Stock")

# Crypto Section
if investment_type == "Crypto":
    crypto_ticker = st.sidebar.text_input("Enter Crypto Ticker (e.g., BTC-USD, ETH-USD)")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, step=100.0)
    
    if st.sidebar.button("Add Crypto Investment"):
        if crypto_ticker and investment_amount > 0:
            crypto_data = fetch_crypto_data(crypto_ticker, start_date, end_date)
            if not crypto_data.empty:
                current_price = crypto_data["Close"].iloc[-1]
                try:
                    c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue) VALUES (?, ?, ?, ?)",
                              ("Crypto", crypto_ticker, investment_amount, investment_amount * (current_price / crypto_data["Close"].iloc[0])))
                    conn.commit()
                    st.sidebar.success(f"Added {crypto_ticker} investment successfully!")
                except sqlite3.Error as e:
                    st.error(f"Failed to add crypto investment: {e}")
    
    if crypto_ticker:
        crypto_data = fetch_crypto_data(crypto_ticker, start_date, end_date)
        if not crypto_data.empty:
            st.subheader(f"{crypto_ticker} Price")
            st.line_chart(crypto_data["Close"], use_container_width=True)
        else:
            st.error("Failed to retrieve data for the given crypto ticker.")
    
    display_portfolio_overview("Crypto")

# Manual Input for Other Investments
if investment_type == "Private Placements":
    company = st.sidebar.text_input("Company")
    industry = st.sidebar.text_input("Industry (i.e. subsector)")
    revenue = st.sidebar.number_input("Revenue (MRQ) ($)", min_value=0.0, step=100.0)
    valuation = st.sidebar.number_input("Valuation ($)", min_value=0.0, step=100.0)
    investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=0.0, step=100.0)
    fund_raise = st.sidebar.number_input("Fund Raise ($)", min_value=0.0, step=100.0)   
    ownership = st.sidebar.number_input("Ownership (%)", min_value=0.0, max_value=100.0, step=0.1)
    date_invested = st.sidebar.date_input("Date Invested")
    platform = st.sidebar.text_input("Platform")
    repayment = st.sidebar.text_input("Repayment")
    frequency = st.sidebar.text_input("Frequency")
    total_return = st.sidebar.number_input("Total Return ($)", min_value=0.0, step=100.0)
    expected_return = st.sidebar.number_input("Expected Return (%)", min_value=0.0, max_value=100.0, step=0.1)
    exit_date = st.sidebar.date_input("Exit Date")
   
    if st.sidebar.button("Add Investment"):
        if company and investment_amount > 0:
            additional_info = f"Industry: {industry}, Revenue: {revenue}, Valuation: {valuation}, ownership: {ownership}, date_invested: {date_invested}, platform: {platform}, repayment: {repayment}, frequency: {frequency}, total_return: {total_return}, exit_date: {exit_date}"            
            try:
                c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue, AdditionalInfo) VALUES (?, ?, ?, ?, ?)",
                          ("Private Placements", company, revenue, valuation, additional_info))
                conn.commit()
                st.sidebar.success(f"Added {company} investment successfully!")
            except sqlite3.Error as e:
                st.error(f"Failed to add private placement investment: {e}")
    
    display_portfolio_overview("Private Placements")

# Startups Section
if investment_type == "Startups":
    pitch_date = st.sidebar.date_input("Pitch Date")
    company = st.sidebar.text_input("Company")
    industry = st.sidebar.text_input("Industry (i.e. subsector)")
    founders = st.sidebar.text_input("Founder(s)")
    website = st.sidebar.text_input("Website")
    email = st.sidebar.text_input("Email")
    fund_raise = st.sidebar.number_input("Fund Raise ($)", min_value=0.0, step=100.0)
    pitch_summary = st.sidebar.text_area("Pitch Summary")
    views = st.sidebar.number_input("Views", min_value=0)
    valuation = st.sidebar.number_input("Valuation ($)", min_value=0.0, step=100.0)
    amount_invested = st.sidebar.number_input("Amount Invested ($)", min_value=0.0, step=100.0)
    ownership = st.sidebar.number_input("Ownership (%)", min_value=0.0, max_value=100.0, step=0.1)
    date_invested = st.sidebar.date_input("Date Invested")
    method = st.sidebar.text_input("Method")
    exit_date = st.sidebar.date_input("Exit Date")

    if st.sidebar.button("Add Startups Investment"):
        if company and amount_invested > 0:
            additional_info = f"Industry: {industry}, Revenue: {revenue}, Valuation: {valuation}, website: {website}, email: {email}, fund_raise: {fund_raise}, pitch_summary: {pitch_summary}"
            try:
                c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue, AdditionalInfo) VALUES (?, ?, ?, ?, ?)",
                          ("Startups", company, revenue, valuation, additional_info))
                conn.commit()
                st.sidebar.success(f"Added {company} investment successfully!")
            except sqlite3.Error as e:
                st.error(f"Failed to add brick & mortar investment: {e}")
    
    display_portfolio_overview("Startups")

# Brick & Mortar Section
if investment_type == "Brick & Mortar":
    company = st.sidebar.text_input("Company")
    industry = st.sidebar.text_input("Industry (i.e. subsector)")
    revenue = st.sidebar.number_input("Revenue (MRQ) ($)", min_value=0.0, step=100.0)
    valuation = st.sidebar.number_input("Valuation ($)", min_value=0.0, step=100.0)
    investors = st.sidebar.text_input("Investors")
    ibrahim_aliyu = st.sidebar.number_input("Ibrahim Aliyu (%)", min_value=0.0, max_value=100.0, step=0.1)
    faruk_aliyu = st.sidebar.number_input("Faruk Aliyu (%)", min_value=0.0, max_value=100.0, step=0.1)
    akh_capital = st.sidebar.number_input("Akh Capital (%)", min_value=0.0, max_value=100.0, step=0.1)

    if st.sidebar.button("Add Brick & Mortar Investment"):
        if company and revenue > 0:
            additional_info = f"Industry: {industry}, Revenue: {revenue}, Valuation: {valuation}, Investors: {investors}, Ibrahim Aliyu: {ibrahim_aliyu}, Faruk Aliyu: {faruk_aliyu}, Akh Capital: {akh_capital}"
            try:
                c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue, AdditionalInfo) VALUES (?, ?, ?, ?, ?)",
                          ("Brick & Mortar", company, revenue, valuation, additional_info))
                conn.commit()
                st.sidebar.success(f"Added {company} investment successfully!")
            except sqlite3.Error as e:
                st.error(f"Failed to add brick & mortar investment: {e}")
    
    display_portfolio_overview("Brick & Mortar")

# FX Section
if investment_type == "FX":
    currency = st.sidebar.text_input("Currency")
    exchange_rate = st.sidebar.number_input("Exchange Rate (current)", min_value=0.0, step=0.01)
    entry_price = st.sidebar.number_input("Entry Price", min_value=0.0, step=0.01)
    entry_date = st.sidebar.date_input("Entry Date")
    target_exit = st.sidebar.number_input("Target Exit", min_value=0.0, step=0.01)

    if st.sidebar.button("Add FX Investment"):
        if currency and exchange_rate > 0:
            additional_info = f"Entry Price: {entry_price}, Entry Date: {entry_date}, Target Exit: {target_exit}"
            try:
                c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue, AdditionalInfo) VALUES (?, ?, ?, ?, ?)",
                          ("FX", currency, entry_price, exchange_rate, additional_info))
                conn.commit()
                st.sidebar.success(f"Added {currency} investment successfully!")
            except sqlite3.Error as e:
                st.error(f"Failed to add FX investment: {e}")
    
    display_portfolio_overview("FX")

# Mudarabah Section
if investment_type == "Mudarabah":
    company = st.sidebar.text_input("Company")
    industry = st.sidebar.text_input("Industry (i.e. subsector)")
    fund_raise = st.sidebar.number_input("Fund Raise ($)", min_value=0.0, step=100.0)
    investors = st.sidebar.text_input("Investors")
    amount_invested = st.sidebar.number_input("Amount Invested ($)", min_value=0.0, step=100.0)
    akh_ownership = st.sidebar.number_input("Akh Ownership (%)", min_value=0.0, max_value=100.0, step=0.1)
    date_invested = st.sidebar.date_input("Date Invested")
    repayment = st.sidebar.text_input("Repayment")
    frequency = st.sidebar.text_input("Frequency")
    total_return = st.sidebar.number_input("Total Return ($)", min_value=0.0, step=100.0)
    exit_date = st.sidebar.date_input("Exit Date")

    if st.sidebar.button("Add Mudarabah Investment"):
        if company and amount_invested > 0:
            additional_info = f"Industry: {industry}, Fund Raise: {fund_raise}, Investors: {investors}, Akh Ownership: {akh_ownership}, Date Invested: {date_invested}, Repayment: {repayment}, Frequency: {frequency}, Total Return: {total_return}, Exit Date: {exit_date}"
            try:
                c.execute("INSERT INTO portfolio (Type, Ticker, AmountInvested, CurrentValue, AdditionalInfo) VALUES (?, ?, ?, ?, ?)",
                          ("Mudarabah", company, amount_invested, total_return, additional_info))
                conn.commit()
                st.sidebar.success(f"Added {company} investment successfully!")
            except sqlite3.Error as e:
                st.error(f"Failed to add mudarabah investment: {e}")
    
    display_portfolio_overview("Mudarabah")

# Portfolio Overview
if investment_type == "Portfolio Overview":
    display_portfolio_overview()

# Close database connection
conn.close()