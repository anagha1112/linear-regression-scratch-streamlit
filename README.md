# 📈 Linear Regression (Scratch Implementation)

This Streamlit app provides a simple, interactive way to explore and apply linear regression from scratch (no external ML libraries used). Users can upload a dataset, select features and target variables, adjust learning parameters, and view results.

## 🔧 Features

- Upload CSV datasets
- Select features (`X`) and target (`y`) for regression
- Configure learning rate and number of iterations
- View weights, bias, MSE, and R² Score
- Visualize regression line (for single-feature models)
- Download predictions as CSV

## 🚀 How to Run

### Option 1: Run on [Streamlit Cloud](https://streamlit.io/cloud)
1. Fork this repo or upload your app to your own GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Click **"New app"**, select your repo, branch, and `app.py` file.
4. Deploy and enjoy!

### Option 2: Run Locally
Make sure you have Python and pip installed.

```bash
git clone https://github.com/anagha1112/linear-regression-scratch-streamlit
cd linear-regression-scratch-streamlit
pip install -r requirements.txt
streamlit run app.py
```

## 📂 File Structure
``` bash
.
├── app.py               # Streamlit app file
├── requirements.txt     # Dependencies
└── README.md            # This file
```
## 📸 Screenshot

![App Screenshot](images/Screenshot%20(80).png)
![App Screenshot](images/Screenshot%20(81).png)
![App Screenshot](images/Screenshot%20(82).png)
![App Screenshot](images/Screenshot%20(83).png)



## 💡 Notes

- This app uses a custom LinearRegression class built from scratch using NumPy.

- Ideal for learning purposes or demonstrating the fundamentals of gradient descent and regression.