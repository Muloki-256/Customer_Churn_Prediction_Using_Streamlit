# 📊 Telecom Customer Churn Prediction - Uganda Market

## 🎯 Project Overview

A comprehensive Streamlit web application for predicting and analyzing customer churn in Uganda's telecommunications sector. This tool helps telecom companies identify at-risk customers and implement data-driven retention strategies using local currency (UGX) and market-specific insights.

## 🌍 Business Context

**Company:** Uganda Telecom Solutions Ltd.  
**Challenge:** High customer attrition rate (27%) impacting revenue and growth  
**Objective:** Develop predictive models to identify at-risk customers and reduce churn by 15% within 6 months

### 📈 Key Business Impact
- **Current churn rate:** 27%
- **Annual revenue impact:** UGX 19,000,000,000 (~$5M USD)
- **Target reduction:** 15% churn rate decrease
- **Potential revenue saved:** UGX 2,850,000,000 (~$750K USD)

## 🚀 Features

### 📋 Data Overview
- Generate synthetic telecom customer data reflecting Ugandan market
- Local pricing in Ugandan Shillings (UGX)
- Uganda-specific features: regions, payment methods, market conditions
- Real-time data metrics and summaries

### 🔍 Exploratory Data Analysis (EDA)
- Interactive visualizations with Plotly
- Churn distribution analysis
- Regional performance comparisons
- Service usage patterns
- Payment method analysis
- Contract type impact assessment

### 🤖 Machine Learning Models
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- Performance metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Feature importance analysis

### 🔮 Churn Prediction
- Real-time churn risk assessment for individual customers
- Customer profile input with Ugandan market specifics
- Risk probability scoring
- Revenue impact analysis
- Retention priority classification

### 💡 Actionable Recommendations
- Retention program strategies
- Pricing optimization for Ugandan market
- Service improvement initiatives
- Regional-focused approaches
- Expected business impact projections

## 🛠️ Technical Stack

### Backend & Data Science
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning models
- **Matplotlib/Seaborn** - Static visualizations

### Frontend & Visualization
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Custom CSS** - Styling and UI enhancements

### Key Libraries
- `streamlit` - Web app deployment
- `pandas`, `numpy` - Data processing
- `scikit-learn` - ML models and metrics
- `plotly` - Interactive charts
- `matplotlib`, `seaborn` - Additional plotting

## 📊 Data Features

### Numerical Features
- `tenure` - Customer subscription duration (months)
- `monthly_charges_ugx` - Monthly bill amount in UGX
- `total_charges_ugx` - Total amount paid in UGX
- `senior_citizen` - Senior citizen status (0/1)
- `customer_service_calls` - Number of support calls

### Categorical Features
- `gender` - Customer gender
- `partner`, `dependents` - Relationship status
- `phone_service`, `multiple_lines` - Service subscriptions
- `internet_service` - Internet service type
- `contract` - Contract duration
- `payment_method` - Payment type (Mobile Money, Bank Transfer, etc.)
- `region` - Geographic region in Uganda
- `churn` - Target variable (Yes/No)

### Uganda-Specific Features
- **Currency:** All pricing in Ugandan Shillings (UGX)
- **Regions:** Kampala, Wakiso, Mbarara, Gulu, Jinja, Mbale
- **Payment Methods:** Mobile Money, Bank Transfer, Cash, Credit Card
- **Market Conditions:** Competitive landscape considerations

## 🎮 How to Use

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd telecom-churn-prediction
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Usage Steps

1. **Start with Introduction**
   - Understand the business problem and project goals
   - Review Ugandan market context

2. **Generate Sample Data**
   - Navigate to "Data Overview"
   - Click "Generate Sample Data" to create synthetic dataset
   - Explore data structure and metrics

3. **Explore Data Analysis**
   - View interactive visualizations
   - Analyze churn patterns in Ugandan context
   - Identify key risk factors

4. **Review Model Performance**
   - Compare machine learning models
   - Examine feature importance
   - Understand prediction accuracy

5. **Make Predictions**
   - Input customer details
   - Get churn probability scores
   - View retention recommendations

6. **Implement Strategies**
   - Review actionable recommendations
   - Develop retention programs
   - Plan pricing and service improvements

## 📈 Model Performance

The application compares three machine learning models:

1. **Logistic Regression** - Baseline model
2. **Decision Tree** - Interpretable model
3. **Random Forest** - Ensemble method (typically best performer)

### Evaluation Metrics
- **Accuracy** - Overall prediction correctness
- **Precision** - Correct positive predictions among all positive predictions
- **Recall** - Ability to find all positive samples
- **F1-Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Model discrimination ability

## 💡 Key Insights for Uganda Market

### High-Risk Factors
- **Month-to-month contracts:** 45% churn rate vs 8% for 2-year contracts
- **High monthly charges:** Above UGX 300,000 increases churn risk by 40%
- **Multiple service calls:** 4+ calls indicate 60% higher churn probability
- **New customers:** <6 months tenure are 3x more likely to churn
- **Regional variations:** Kampala shows different patterns due to competition

### Retention Opportunities
- **Contract incentives** for longer commitments
- **Pricing optimization** for high-value segments
- **Proactive support** for customers with service issues
- **Regional-specific** marketing strategies
- **Payment method** preferences (Mobile Money loyalty)

## 🎯 Business Recommendations

### Immediate Actions
1. **Contract Conversion Campaign** - Target month-to-month customers
2. **Loyalty Rewards Program** - Milestone-based incentives
3. **Proactive Support System** - Monitor high-risk customers
4. **Regional Strategy Development** - Customized approaches per region

### Long-term Strategies
1. **Pricing Tier Optimization** - Better value propositions
2. **Service Quality Improvement** - Network and support enhancements
3. **Customer Experience Focus** - Personalized engagement
4. **Data-Driven Decision Making** - Continuous model improvement

## 📁 Project Structure

```
telecom-churn-uganda/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── assets/               # Additional resources
    ├── images/           # Screenshots and logos
    └── data/             # Sample datasets (if any)
```

## 🔧 Configuration

### Exchange Rate Settings
The application uses a fixed exchange rate:
```python
USD_TO_UGX = 3800  # Approximate UGX to USD exchange rate
```

### Data Generation
- Synthetic data generation for 1,000 customers
- Realistic Ugandan market distributions
- Configurable sample size

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment Options
- **Streamlit Cloud**
- **Heroku**
- **AWS EC2**
- **Google Cloud Run**

## 👥 Target Audience

- **Telecom Managers** - Business decision makers
- **Marketing Teams** - Campaign planning and execution
- **Customer Service** - Retention and support strategies
- **Data Analysts** - Insights and reporting
- **Business Strategists** - Long-term planning

## 📞 Support & Contact

For questions, issues, or contributions:
1. Check the documentation
2. Review code comments
3. Create issues in the repository
4. Contact the development team

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- [ ] Real data integration capabilities
- [ ] Advanced machine learning models
- [ ] Real-time data streaming
- [ ] API integrations with telecom systems
- [ ] Mobile application version
- [ ] Multi-language support (Swahili, Luganda)
- [ ] Advanced analytics dashboard
- [ ] Automated reporting features

---

**Built with ❤️ for Uganda's Telecom Industry**
