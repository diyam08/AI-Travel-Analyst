# AI-Travel-Analyst
Project Overview:

The AI Travel Analyst is an end-to-end Machine Learning application designed to evaluate domestic flight data, uncover critical travel insights, and perform real-time price estimation. It is built mainly using Python, Scikit-Learn, Streamlit and Plotly. The system features a trained Random Forest Regressor and a 4-tab interactive web dashboard deployed using streamlit.

Problem Statement:

Flight prices change constantly. Factors like how early you book, which airline you choose, travel class, and flight time all affect the final fare. This uncertainty makes it tricky for travelers to know when to buy.
This project solves this by using data science and machine learning to:
1.Uncover price patterns(Finds out what factors make tickets expensive or cheap.)
2.Predict ticket prices: (Estimates fares instantly based on user inputs.)
3.Search within budget: (Helps budget conscious travelers find flights they can afford.)

Installation Instructions:

1.Download the code:
git clone https://github.com/your-username/ai-travel-analyst.git
cd ai-travel-analyst
2.Set Up Environment & Install Dependencies(streamlit, scikit-learn, pandas, plotly, joblib):
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3.Launch the Web App:
streamlit run app.py

Dataset Used:

Source File: cleaned_flight_data.csv 
Main Features:
Flight Info: Airline Carrier, Source City, Destination City, Travel Class.
Timing Info: Departure Time, Arrival Time, Date of Journey, Advance Booking Days.
Flight Specs: Total Stops, Route Distance in km, Passenger Count.
Target Variable: Flight Ticket Price (Price in INR ₹).

Methodology:

Raw CSV ➔ Data Cleaning and Pre-processing ➔ Create visualisations ➔ Feature Engineering ➔ Model Training ➔ Model Testing ➔ Web Deployment

Technologies Used:

1.Core Programming: Python 3.10+
2.Data Processing: Pandas, NumPy
3.Machine Learning: Scikit-Learn, Joblib
4.Data Visualization: Plotly, Matplotlib
5.Web Dashboard: Streamlit
6.Deployment & Hosting: Streamlit Cloud, GitHub

Results:

Model Evaluation Metrics (Testing Set):
R^2 Performance Score: 0.6152 (Thus the model displays 61.52% of overall price variance across complex routes).
Mean Absolute Error : ₹16,097.84
Root Mean Squared Error : ₹48,408.95

Challenges Faced:

1.Memory Limits while deployment: The pickle files while uploading to GitHub exceeded the basic cloud memory limits(25MB). Thus a separate code to compress it was used before uploading.
2.Overfitting & Non-Linearity in Dynamic Pricing: Flight dares sometimes spike non-linearly near departure dates. Thus there will be a noticeable variance between training and testing accuracy. 

Future Improvements:

1.Incorporate log-target transformations and hyperparameter regularization to reduce the variance in accuracy between training and testing results.
2.Integrate live aviation API feeds for real-time ticket fetching.
3. While Flight Recommendation System has been implemented other features like Cheapest Booking Time Analysis and Flight Price Forecasting can be done in the future.



