**British Airways Data Science Job Simulation**
A machine learning project completed as part of the British Airways Data Science Virtual Experience Programme on Forage.

**Overview**
British Airways wanted to understand what drives a customer to complete a flight booking. This project builds a predictive model using a Random Forest classifier trained on 50,000 booking records, and identifies which customer and flight characteristics have the most influence on whether a booking is completed.

**What I Did**
**Exploratory data analysis** - examined the dataset structure, identified a significant class imbalance (85% of customers do not complete a booking), and explored the distribution of key variables including booking origin, flight route, and purchase lead time.
**Feature engineering **- created five new features to improve model performance:
total_extras - combined score of baggage, seat, and meal add-ons selected, as a proxy for customer intent
lead_bucket - grouped purchase lead time into behavioural categories (last minute, short notice, planned, far advance)
is_convenient_hour - binary flag for flights departing between 06:00–21:00
is_weekend - binary flag for Saturday/Sunday departures
route_popularity - how frequently each route appears in the dataset

**Model training** - trained a Random Forest with class_weight='balanced' to handle the 85/15 class imbalance, preventing the model from simply predicting "not booked" for every customer.
**Evaluation** - used stratified 5-fold cross-validation to get a reliable performance estimate across the full dataset.
The model consistently scored ROC-AUC of 0.76 (±0.004) across all five folds, indicating stable generalisation rather than overfitting to any one subset of the data.

**Key Findings**
Booking origin is the strongest predictor, accounting for 43% of the model's predictive power. Customers from different countries complete bookings at very different rates, a meaningful insight for targeted marketing strategy.
Route popularity and length of stay are the next most important features (14% and 12% respectively). Longer trips on well-travelled routes are significantly more likely to result in a completed booking.
Purchase lead time and flight hour contribute moderate signal. How far in advance a customer books, and the time of day of the flight, both influence completion likelihood.
Add-on selections signal intent. The engineered total_extras feature ranked above individual add-on flags, suggesting that the cumulative number of extras selected is a better signal of commitment than any single choice.

**Tools & Libraries**
Python · pandas · scikit-learn · matplotlib · Random Forest · Stratified K-Fold Cross-Validation
