from flask import Flask, request, session, flash
from flask import render_template, redirect, url_for
from werkzeug.utils import secure_filename
from database import engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import re
import os
import pickle
import utils
import train_models as tm

app = Flask(__name__)
app.secret_key = "king_AS"
UPLOAD_PATH = "individual_stocks_5yr"
app.config['UPLOAD_FLODER'] = UPLOAD_PATH

pol = pickle.load(
    open('model/poly.pkl', 'rb'))
regresso = pickle.load(
    open('model/regressor.pkl', 'rb'))

all_files = utils.read_all_stock_files('individual_stocks_5yr')


def model(filename):
    nifty50 = pd.read_csv("individual_stocks_5yr/"+filename+"_5y.csv")
    nifty50['date'] = pd.to_datetime(nifty50['date'])
    nifty50['Year'] = nifty50['date'].dt.year
    nifty50['Month'] = nifty50['date'].dt.month
    nifty50['Day'] = nifty50['date'].dt.day
    nifty50['dayOfWeek'] = nifty50['date'].dt.day_name()
    nifty50['WeekOfYear'] = nifty50['date'].dt.isocalendar().week
    close = nifty50['close']
    X = nifty50[['Day', 'Month', 'Year']]
    poly2 = PolynomialFeatures(degree=2)
    X_poly = poly2.fit_transform(X.values)
    regressor = RandomForestRegressor(n_estimators=15, random_state=0)
    regressor.fit(X_poly, close)
    return poly2, regressor


def perform_training(stock_name, df, models_list):
    all_colors = {'SVR_linear': '#FF9EDD',
                  'SVR_poly': '#FFFD7F',
                  'SVR_rbf': '#FFA646',
                  'linear_regression': '#CC2A1E',
                  'random_forests': '#8F0099',
                  'KNN': '#CCAB43',
                  'elastic_net': '#CFAC43',
                  'DT': '#85CC43',
                  'LSTM_model': '#CC7674'}

    print(df.head())
    dates, prices, ml_models_outputs, prediction_date, test_price = tm.train_predict_plot(
        stock_name, df, models_list)
    origdates = dates
    if len(dates) > 20:
        dates = dates[-20:]
        prices = prices[-20:]

    all_data = []
    all_data.append((prices, 'false', 'Data', '#000000'))
    for model_output in ml_models_outputs:
        if len(origdates) > 20:
            all_data.append(
                (((ml_models_outputs[model_output])[0])[-20:], "true", model_output, all_colors[model_output]))
        else:
            all_data.append(
                (((ml_models_outputs[model_output])[0]), "true", model_output, all_colors[model_output]))

    all_prediction_data = []
    all_test_evaluations = []
    all_prediction_data.append(("Original", test_price))
    for model_output in ml_models_outputs:
        all_prediction_data.append(
            (model_output, (ml_models_outputs[model_output])[1]))
        all_test_evaluations.append(
            (model_output, (ml_models_outputs[model_output])[2]))

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations


@app.route("/")
def first():
    return render_template("first.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/future")
def future():
    return render_template("future.html")


@app.route("/backupLogin", methods=['GET', 'POST'])
def backupLogin():
    if request.method == 'POST' and 'uname' in request.form and 'psw' in request.form:
        username = request.form['uname']
        password = request.form['psw']
        query = engine.execute(
            f"Select * From `users` Where user_id = '{username}' And password = '{password}'")
        account = query.fetchone()
        print(f"{username} \n{password}")
        if account:
            session['loggedin'] = True
            session['id'] = account['email']
            session['username'] = request.form['uname']
            return render_template('first.html')
        else:
            flash("Invalid Credentials!")
    return render_template("backupLogin.html")


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'uname' in request.form and 'psw' in request.form:
        username = request.form['uname']
        password = request.form['psw']
        query = engine.execute(
            f"Select * From `users` Where user_id = '{username}' And password = '{password}'")
        account = query.fetchone()
        print(f"{username} \n{password}")
        if account:
            session['loggedin'] = True
            session['id'] = account['email']
            session['username'] = request.form['uname']
            return render_template('first.html')
        else:
            flash("Invalid Credentials!")
    return render_template("login.html")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST' and 'uname' in request.form and 'email' in request.form and 'psw' in request.form:
        username = request.form['uname']
        password = request.form['psw']
        email = request.form['email']
        repeated = request.form['psw-repeat']
        query = engine.execute(
            f"SELECT * FROM users WHERE user_id = '{username}'")
        account = query.fetchone()

        if account:
            flash('Account already exists !')
            return render_template('login.html')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers !')
        else:
            if repeated == password:
                engine.execute(
                    f"INSERT INTO users VALUES ('{username}', '{email}', '{password}')")
                flash('You have successfully registered !')
                return render_template('login.html')
            else:
                flash("Both Passwords should match")
    elif request.method == 'POST':
        flash('Please fill out the form !')
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


def read_csv_upload(filename):
    df = pd.read_csv(UPLOAD_PATH+'/'+filename, encoding='unicode_escape')
    return df


@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        filename = secure_filename(dataset.filename)
        dataset.save(os.path.join(app.config['UPLOAD_FLODER'], filename))
        df = read_csv_upload(filename=filename)
        return render_template("preview.html", df_view=df)


@app.route('/landing_function')
# ‘/’ URL is bound with hello_world() function.
def landing_function():
    all_files = utils.read_all_stock_files('individual_stocks_5yr')
    # df = all_files['A']
    # # df = pd.read_csv('GOOG_30_days.csv')
    # all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data = perform_training('A', df, ['SVR_linear'])
    stock_files = list(all_files.keys())

    return render_template('index.html', show_results="false", stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[],
                           prediction_date="", dates=[], all_data=[], len=len([]))


@app.route('/process', methods=['POST'])
def process():

    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')

    all_files = utils.read_all_stock_files('individual_stocks_5yr')
    df = all_files[str(stock_file_name)]
    # df = pd.read_csv('GOOG_30_days.csv')
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = perform_training(
        str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())

    return render_template('index.html', all_test_evaluations=all_test_evaluations, show_results="true", stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data))


@app.route('/predict', methods=['POST'])
def predict():
    # int_feature = [x for x in request.form.values()]
    int_feature = request.form["Date"].split("-")
    int_feature.reverse()
    # int_feature.append(request.form["Indices"])
    # print(int_feature)
    pol, regresso = model(filename=request.form["Indices"])
    final_features = [np.array(int_feature)]
    Total_infections = pol.transform(final_features)
    prediction = regresso.predict(Total_infections)
    pred = format(int(prediction[0]))

    return f"Closing Price = {pred}"


if __name__ == "__main__":
    app.run(debug=True)
