import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from streamlit_navigation_bar import st_navbar
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# page = st_navbar(["Demand", "Supply","Projection"])
# Custom CSS for fonts
page = st.sidebar.radio("Pages", ["Demand", "Supply"])
demand_data = pd.read_csv("GSM_demandbysector.csv")
print(demand_data.head())
demand_data['Power Generation']=demand_data['Power Generation'].str.replace(',', "").astype(float)
demand_data['Total']=demand_data['Total'].str.replace(',', "").astype(float)
demand_data[['Power Generation','Industry','NGV','GSP','Others','Total']] = demand_data[['Power Generation','Industry','NGV','GSP','Others','Total']].replace('-', 0).astype(float)
demand_data['Date'] = pd.to_datetime(demand_data['Date'])

# Page content
if page == "Demand":
    st.title("Welcome to the Demand page")
    st.write("This will simulate the demand of PTT")
    
    st.write("""
    ## Our prediction of demand

    """)
    actualdemandtotal = pd.read_csv("Total_Demand_y_test.csv")
    predictdemandtotal = pd.read_csv("Total_Demand_y_pred.csv")
    
# Select the last 7 rows of the dataset
    actual_totallast7 = actualdemandtotal.tail(7)  # You can use tail(7) instead of slicing manually
    predict_totallast7 = predictdemandtotal.tail(7)
    st.write("We predict demand of the next day:" ,(predict_totallast7.tail(1))['Date'].values[0])
    st.write("\nThe predicted demand will be",(predict_totallast7.tail(1))['Total_Demand'].values[0], " MMSCFD")
    st.write("Actual demand is:" ,(actual_totallast7.tail(1))['Total_Demand'].values[0],"MMSCFD")
    predictdemand_nextday  = (predict_totallast7.tail(1))['Total_Demand'].values[0]
    # Convert 'Date' to datetime if it's not already in that format
    # Convert 'Date' to datetime if it's not already in that format
    actual_totallast7['Date'] = pd.to_datetime(actual_totallast7['Date'], errors='coerce')
    predict_totallast7['Date'] = pd.to_datetime(predict_totallast7['Date'], errors='coerce')
    # Check if the conversion worked
    # print(actual_totallast7)
    actualdemandtotal = pd.read_csv("Total_Demand_y_test.csv")
    predictdemandtotal = pd.read_csv("Total_Demand_y_pred.csv")

    # Select the last 7 rows of the datasets
    actual_totallast7 = actualdemandtotal.tail(7)
    predict_totallast7 = predictdemandtotal.tail(7)

    # Convert 'Date' to datetime if it's not already in that format
    actual_totallast7['Date'] = pd.to_datetime(actual_totallast7['Date'], errors='coerce')
    predict_totallast7['Date'] = pd.to_datetime(predict_totallast7['Date'], errors='coerce')

    # Create DataFrame for plotting
    # Separate the actual demand (first 6 days) and the prediction (day 7)
    actual_data = actual_totallast7.head(6)  # First 6 days of actual data
    prediction_data = predict_totallast7.tail(1)  # Last day (prediction)

    plt.plot(actual_totallast7['Date'], actual_totallast7['Total_Demand'], 'bo-', label='Actual Demand')

# Plot the predicted demand for the 7th day as a red "x"
    plt.plot(predict_totallast7['Date'].iloc[-1], predict_totallast7['Total_Demand'].iloc[-1], 'rx', markersize=10, label='Predicted Demand')

    # Adding labels and title
    plt.title('Actual vs Predicted Demand for the Last 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.tight_layout()  # Adjusts plot to ensure it fits well within the window
    st.pyplot(plt)  # If using Streamlit
    data = pd.DataFrame({
        'Spot': actual_totallast7['Date'],
        'Total Actual Demand': actual_totallast7['Total_Demand'].values,
        'Total Predicted Demand': predict_totallast7['Total_Demand'].values
    })

    # Display the line chart
    st.line_chart(data.set_index('Spot'))
    category = ['PowerGeneration','Industry' ,'GSP','NGV']
    listnextday =[]
    for i in category:
        st.write(f"""
        #### Our Prediction for demand of the {i}

        """)
        actualdemandtotal = pd.read_csv(i+"_Demand_y_test.csv")
        predictdemandtotal = pd.read_csv(i+"_Demand_y_pred.csv")

        # Select the last 7 rows of the datasets
        actual_totallast7 = actualdemandtotal.tail(7)
        predict_totallast7 = predictdemandtotal.tail(7)

        # Convert 'Date' columns to datetime format
        

        # Display the prediction and actual demand for the next day
        datewepredict =predict_totallast7.tail(1)['Date'].values[0]
        st.write("We predict demand for the next day:", (predict_totallast7.tail(1))['Date'].values[0])
        st.write("\nThe **predicted demand** will be", (predict_totallast7.tail(1))[i].values[0], "MMSCFD")
        st.write("**Actual demand** is:", (actual_totallast7.tail(1))[i].values[0], "MMSCFD")
        listnextday.append((predict_totallast7.tail(1))[i].values[0])
        actual_totallast7['Date'] = pd.to_datetime(actual_totallast7['Date'], errors='coerce')
        predict_totallast7['Date'] = pd.to_datetime(predict_totallast7['Date'], errors='coerce')
        # Plot the actual demand (first 6 days) and the predicted demand (7th day)
        plt.figure(figsize=(10, 5))
        plt.plot(actual_totallast7['Date'], actual_totallast7[i], 'bo-', label=f'Actual Demand of {i}')

        # Plot the predicted demand for the 7th day as a single red "x"
        plt.plot(predict_totallast7['Date'].iloc[-1], predict_totallast7[i].iloc[-1], 'rx', markersize=10, label=f'Predicted Demand of {i}')

        # Adding labels and title
        plt.title(f'Actual vs Predicted Demand of {i} for the Last 7 Days')
        plt.xlabel('Date')
        plt.ylabel(f'Total Demand of {i} (MMSCFD)')

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display the legend
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjusts plot to ensure it fits well within the window
        st.pyplot(plt)
        data = pd.DataFrame({
        'Spot': actual_totallast7['Date'],
        f'Total Actual of {i}': actual_totallast7[i].values,
        f'Total Predicted Demand of {i}': predict_totallast7[i].values
        })

    # Display the line chart
        st.line_chart(data.set_index('Spot'))

    st.write(f"""
    #### The dashboard shows overall predicted demand at {datewepredict}

    """)
    labels = ['Power Generation', 'Industry', 'GSP', 'NGV']
    sizes = listnextday
    blue_colors = ['#5D8AA8', '#1E3A5F', '#2C7B9A', '#4A90B8', '#7BB9D6']

    fig, ax = plt.subplots(figsize=(50,50))

    # Create a pie chart with a 'hole' in the middle to form a donut
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',  # Show percentages with 1 decimal place
        startangle=90, 
        wedgeprops={'width': 0.5},  # This creates the hole in the middle
        colors=blue_colors  # Apply the blue tones
    )
    for autotext in autotexts:
        autotext.set_fontsize(80) 
    for text in texts:
        text.set_fontsize(80)
    ax.set_title('Gas Demand from each sector', fontsize=100, color='navy')
    st.pyplot(fig)
    st.write("""
    ## The dashboard shows demand for each sector

    """)
    demand_date = st.date_input("Select the date for demand")
    st.write("Your demand date is:", demand_date)

    demand_date = pd.to_datetime(demand_date)
    #find column same demand date
    selected_data = demand_data[demand_data['Date'] == demand_date]
    # print(selected_data)
    # print(demand_data['Date'] == demand_date)
    if not selected_data.empty:
        st.subheader(f"Summary for {demand_date.strftime('%Y-%m-%d')}")
        col1, col2 = st.columns(2)

        # Column 1: Metrics summary
        with col1:
        # Display metrics
            st.metric("Power Generation (MMSCFD)", f"{selected_data['Power Generation'].values[0]:,.2f}")
            st.metric("Industry (MMSCFD)", f"{selected_data['Industry'].values[0]:,.2f}")
            st.metric("NGV (MMSCFD)", f"{selected_data['NGV'].values[0]:,.2f}")
            st.metric("GSP (MMSCFD)", f"{selected_data['GSP'].values[0]:,.2f}")
            st.metric("Others (MMSCFD)", f"{selected_data['Others'].values[0]:,.2f}")
            st.metric("Total (MMSCFD)", f"{selected_data['Total'].values[0]:,.2f}")
        with col2:
        # Create a pie chart
            labels = ['Power Generation', 'Industry', 'NGV', 'GSP', 'Others']
            sizes = [selected_data['Power Generation'].values[0], selected_data['Industry'].values[0], selected_data['NGV'].values[0], selected_data['GSP'].values[0], selected_data['Others'].values[0]]
            blue_colors = ['#5D8AA8', '#1E3A5F', '#2C7B9A', '#4A90B8', '#7BB9D6']

            fig, ax = plt.subplots(figsize=(50,50))

            # Create a pie chart with a 'hole' in the middle to form a donut
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%',  # Show percentages with 1 decimal place
                startangle=90, 
                wedgeprops={'width': 0.5},  # This creates the hole in the middle
                colors=blue_colors  # Apply the blue tones
            )
            for autotext in autotexts:
                autotext.set_fontsize(80) 
            for text in texts:
                text.set_fontsize(80)
            ax.set_title('Gas Demand', fontsize=100, color='navy')
            st.pyplot(fig)
    else:
        st.write("No data at", demand_date)


    st.write("""
    ## Trend of total demand

    """)
    # demand_trend_start = st.date_input("The start date")
    # demand_trend_end = st.date_input("The end date")
    # Preprocess the demand_data to extract year and month
    demand_data['Year'] = demand_data['Date'].dt.year
    demand_data['Month'] = demand_data['Date'].dt.strftime('%B')

    # Filter data by year
    demand_data_2022 = demand_data[demand_data['Year'] == 2022]
    demand_data_2023 = demand_data[demand_data['Year'] == 2023]
    demand_data_2024 = demand_data[demand_data['Year'] == 2024]

    # Checkbox to show trends
    show_all =st.checkbox('Show the trend of demand in 2022, 2023, and 2024', value=True)
    show_2022 = st.checkbox('Show the trend of demand in 2022')
    show_2023 = st.checkbox('Show the trend of demand in 2023')
    show_2024 = st.checkbox('Show the trend of demand in 2024')
    # st.write("Trend Demand Data for 2022", demand_data_2022)
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

        # Group and aggregate total demand by month
    monthly_total_2022 = (
        demand_data_2022.groupby('Month')['Total']
        .sum()
        .reindex(month_order, fill_value=0)
    )
    monthly_total_2023 = (
        demand_data_2023.groupby('Month')['Total']
        .sum()
        .reindex(month_order, fill_value=0)
    )
    monthly_total_2024 = (
        demand_data_2024.groupby('Month')['Total']
        .sum()
        .reindex(month_order, fill_value=0)
    )

        # Combine into a single DataFrame
    if show_all:
        # st.write("Trend Demand Data for 2022", demand_data_2022)
        data = pd.DataFrame({
                'Spot': month_order,
                '2022': monthly_total_2022.values,
                '2023': monthly_total_2023.values,
                '2024': monthly_total_2024.values
        })
        data['Spot'] = pd.Categorical(data['Spot'], categories=month_order, ordered=True)
        data = data.sort_values('Spot')
        data = data.set_index('Spot')
        st.line_chart(data,color=["#A6AEBF","#9694FF","#77CDFF"] )
    if show_2022:
        # st.write("Trend Demand Data for 2022", demand_data_2022)
        data = pd.DataFrame({
                'Spot': month_order,
                '2022': monthly_total_2022.values,
                
        })
        data['Spot'] = pd.Categorical(data['Spot'], categories=month_order, ordered=True)
        data = data.sort_values('Spot')
        data = data.set_index('Spot')
        st.area_chart(data,color=["#A6AEBF"] )
    # Month order for consistent sorting
    if show_2023:
        # st.write("Trend Demand Data for 2022", demand_data_2022)
        data = pd.DataFrame({
                'Spot': month_order,

                '2023': monthly_total_2023.values,

        })
        data['Spot'] = pd.Categorical(data['Spot'], categories=month_order, ordered=True)
        data = data.sort_values('Spot')
        data = data.set_index('Spot')
        st.area_chart(data,color=[ "#9694FF"] )
    if show_2024:
        # st.write("Trend Demand Data for 2022", demand_data_2022)
        data = pd.DataFrame({
                'Spot': month_order,
                '2024': monthly_total_2024.values
        })
        data['Spot'] = pd.Categorical(data['Spot'], categories=month_order, ordered=True)
        data = data.sort_values('Spot')
        data = data.set_index('Spot')
        st.area_chart(data,color=[ "#77CDFF"] )
    instant_change_demand = st.checkbox('There is instant occasion and you want to change the quantity of demand')
    if instant_change_demand:
        predictdemand_nextday = st.number_input("Input the changed demand",value=predictdemand_nextday)
    selling_price = st.number_input("Input the selling price (Bath/MMBtu)", value= 347.4511)
    st.write("""
    ## Total Revenue

    """)
    Totalrevenue =  predictdemand_nextday*selling_price
    st.write("Total Revenue", Totalrevenue, "Bath")

elif page == "Supply":
    st.title("Welcome to the Supply page")
    st.write("This will simulate the supply of PTT")
    JKM_actual = pd.read_csv("JKM_Weekly_y_test.csv")
    JKM_predict = pd.read_csv("JKM_Weekly_y_pred.csv")
    print(JKM_predict.head())
   
    JKM_predict_last10w = JKM_predict.tail(48)
    JKM_actual_last10w = JKM_actual.tail(48)
    # print(JKM_actual.tail(11))
    st.write("We forecast the End Spot LNG price for the upcoming week, concluding on:" ,(JKM_predict_last10w.tail(1))['Date'].values[0])
    st.write("\nThe predicted Spot LNG price will be",(JKM_predict_last10w.tail(1))['JKM_Price'].values[0],"USD/MMBTU")
    st.write("Actual Spot LNG price is:" ,(JKM_actual_last10w.tail(1))['JKM_Price'].values[0],"USD/MMBTU") 
    supply=JKM_predict_last10w.tail(1)['JKM_Price'].values[0]
    JKM_actual['Date'] = pd.to_datetime(JKM_actual['Date'], errors='coerce' )
    JKM_predict['Date'] = pd.to_datetime(JKM_predict['Date'], errors='coerce' )
    JKM_actual['Year'] = JKM_actual['Date'].dt.year
    JKM_actual['Month'] = JKM_actual['Date'].dt.strftime('%B')
    JKM_predict['Year'] = JKM_predict['Date'].dt.year
    JKM_predict['Month'] = JKM_predict['Date'].dt.strftime('%B')
    actual_data = JKM_actual_last10w  # First 6 days of actual data
    prediction_data = JKM_predict_last10w.tail(1)  # Last day (prediction)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(JKM_actual_last10w['Date'], JKM_actual_last10w['JKM_Price'], 'bo-', label='Actual Spot LNG price per week')

# Plot the predicted demand for the 7th day as a red "x"
    ax.plot(JKM_predict_last10w['Date'].iloc[-1], JKM_predict_last10w['JKM_Price'].iloc[-1], 'rx', markersize=10, label='Predicted Spot LNG price per week')

    # Adding labels and title
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Actual vs Predicted Spot LNG price(JKM) for the Last 48 weeks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spot LNG price (USD/MMBTU)')

    # Rotate the x-axis labels for better readability
    # plt.xticks(rotation=90)
    ax.legend()

    # plt.tight_layout()  # Adjusts plot to ensure it fits well within the window
    st.pyplot(plt) 

    st.write("Trend of Spot LNG price since 11/13/2022")
    data = pd.DataFrame({
        'Spot': JKM_actual['Date'],
        'Actual Spot LNG Price (USD/MMBTU)': JKM_actual['JKM_Price'].values,
        'Predict Spot LNG Price (USD/MMBTU)': JKM_predict['JKM_Price'].values
    })

    # Display the line chart
    st.line_chart(data.set_index('Spot'))
    instant_change_supply = st.checkbox('There is instant occasion and you want to change the next 7 days LNG price')
    if instant_change_supply:
        supply= st.number_input("Input the changed Spot LNG Price next 7 days",value=supply)
    shipping_cost = st.number_input("The shipping cost of spot LNG :")  
    st.write("""
    ## Supply from Myanmar

    """)
    st.write("The prediction of Myanmar price will come up soon.")

    st.write("""
    ## Gas production plan

    """)
    gasgulfthaistore = st.number_input("The current quantity of the natural gas store from **Gulf of Thailand** in the storage (M Cubric foot):",value=100)
    pricegasgulf = st.number_input("Price of the natural gas store from **Gulf of Thailand** in the storage (Bath/MMBTU):",value=100)
    gasonshorestore = st.number_input("The current quantity of the natural gas store from **Onshore** in the storage (M Cubric foot):",value=100)
    priceonshore = st.number_input("Price of the natural gas store from **Onshore** in the storage (Bath/MMBTU):",value=100)
    storage_cost = st.number_input("The cost of reservation of the **LD storage** (Bath/MMBTU):",value=16)
    
