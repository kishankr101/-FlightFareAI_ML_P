# -FlightFareAI_ML_P

✈️ FlightFareAI – Indian Domestic Flight Price Predictor

FlightFareAI is a machine learning-powered web app built with Streamlit that predicts Indian domestic flight ticket prices based on key travel factors like airline, route, timing, and class.

It helps travelers estimate fair ticket prices, compare models, and gain insights into what drives airfare variations.

🚀 Live Features
🔮 Price Prediction
Predict ticket prices instantly using ML models
Compare predictions from:
Random Forest (Best Model 🏆)
Decision Tree
Linear Regression

📊 Model Insights
R² Score & MAE comparison
Airline vs Price distribution
Class-based pricing trends
Price distribution visualization

🎯 Smart Recommendations
Displays estimated price range
Provides money-saving tips based on selection

🧠 Machine Learning Models Used
Model	R² Score	MAE

🏆 Random Forest	0.9754	₹2,163
Decision Tree	0.9749	₹2,159
Linear Regression	0.8983	₹4,911

👉 Best Model: Random Forest Regressor

📂 Dataset Information
Type: Indian Domestic Flight Data
Size: 300,153 rows × 12 columns
Target Variable: price

✨ Features Used
Feature	Description
airline	Airline name
source_city	Departure city
destination_city	Arrival city
departure_time	Time of departure
arrival_time	Time of arrival
stops	Number of stops
class	Economy / Business
duration	Flight duration (hours)


🧹 Data Preprocessing
Removed duplicates
Dropped irrelevant columns:
Unnamed: 0, flight, days_left
Applied Label Encoding to categorical variables
Train-test split: 80/20
🏗️ Project Structure
📁 FlightFareAI
│
├── flight_price_predictor.py   # Main Streamlit app
├── Clean_Dataset.csv           # Dataset
├── README.md                   # Project documentation


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/FlightFareAI.git
cd FlightFareAI
2️⃣ Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
3️⃣ Run the app
streamlit run flight_price_predictor.py


🖥️ Usage
Upload or provide the dataset path (Clean_Dataset.csv)
Select:
Airline
Source & Destination
Time
Stops
Class
Duration
Click Predict Ticket Price
View:
Predicted fares
Best estimate

Model comparison chart
📊 Visualizations Included
📈 Model Performance Comparison
✈ Airline vs Price (Boxplot)
💺 Class vs Price (Bar chart)
📉 Price Distribution (Histogram)
💡 Key Insights

Business class tickets are significantly more expensive than Economy
Early morning flights tend to be cheaper
Ticket prices increase with duration and fewer stops
Random Forest captures pricing patterns with high accuracy

🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
ML Library: Scikit-learn
Visualization: Matplotlib, Seaborn
Data Handling: Pandas, NumPy

📌 Future Improvements
🌐 Deploy on Streamlit Cloud / AWS
📅 Include days_left feature for better accuracy
🤖 Add Deep Learning models
📱 Mobile-friendly UI improvements
🔄 Real-time API integration for live fares
