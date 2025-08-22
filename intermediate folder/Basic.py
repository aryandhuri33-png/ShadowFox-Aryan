from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import datetime
import os   
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)


# Define DB model
class CarPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    present_price = db.Column(db.Float)
    kms_driven = db.Column(db.Integer)
    owner = db.Column(db.Integer)
    fuel_type = db.Column(db.String(20))
    seller_type = db.Column(db.String(20))
    transmission = db.Column(db.String(20))
    year = db.Column(db.Integer)
    car_age = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    plot_data = None
    error_message = None

    if request.method == 'POST':
        try:
            # Collect form data
            present_price = request.form.get('present_price')
            kms_driven = request.form.get('kms_driven')
            owner = request.form.get('owner')
            fuel_type = request.form.get('fuel_type')
            seller_type = request.form.get('seller_type')
            transmission = request.form.get('transmission')
            year = request.form.get('year')

            # Validate inputs
            if not all([present_price, kms_driven, owner, fuel_type, seller_type, transmission, year]):
                error_message = "\u26a0\ufe0f Please enter all values before predicting."
                return render_template("index.html", predicted_price=None, plot_data=None, error_message=error_message)

            # Convert inputs
            present_price = float(present_price)
            kms_driven = int(kms_driven)
            owner = int(owner)
            year = int(year)
            car_age = datetime.datetime.now().year - year

            input_df = pd.DataFrame([{
                'Present_Price': present_price,
                'Kms_Driven': kms_driven,
                'Owner': owner,
                'Fuel_Type': fuel_type,
                'Seller_Type': seller_type,
                'Transmission': transmission,
                'car_age': car_age
            }])

            # Predict
            predicted_price = round(model.predict(input_df)[0], 2)

            # Save prediction to DB
            new_entry = CarPrediction(
                present_price=present_price,
                kms_driven=kms_driven,
                owner=owner,
                fuel_type=fuel_type,
                seller_type=seller_type,
                transmission=transmission,
                year=year,
                car_age=car_age,
                predicted_price=predicted_price
            )
            db.session.add(new_entry)
            db.session.commit()

            # Create Plot
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            labels = ['Showroom Price', 'Predicted Price', 'Difference']
            values = [present_price, predicted_price, abs(present_price - predicted_price)]
            colors = ['#007bff', '#28a745', '#dc3545']
            bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='black')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10, weight='bold')

            ax.set_title("Car Price Prediction Breakdown", fontsize=14, weight='bold')
            ax.set_ylabel("Price (Lakhs)")
            ax.set_ylim(0, max(values) + 2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

        except Exception as e:
            error_message = f"An error occurred: {e}"

    return render_template("index.html", predicted_price=predicted_price, plot_data=plot_data, error_message=error_message)

@app.route('/history')
def history():
    all_data = CarPrediction.query.order_by(CarPrediction.timestamp.desc()).all()
    return render_template("history.html", data=all_data)

@app.route('/download_csv')
def download_csv():
    all_data = CarPrediction.query.order_by(CarPrediction.timestamp.desc()).all()
    data = [{
        "ID": d.id,
        "Showroom Price": d.present_price,
        "KMs Driven": d.kms_driven,
        "Owner": d.owner,
        "Fuel Type": d.fuel_type,
        "Seller Type": d.seller_type,
        "Transmission": d.transmission,
        "Year": d.year,
        "Car Age": d.car_age,
        "Predicted Price": d.predicted_price,
        "Timestamp": d.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for d in all_data]

    df = pd.DataFrame(data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='car_predictions.csv'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # ✅ Ensures the DB and tables exist
        db.session.query(CarPrediction).delete()  # 🔥 TEMPORARY: Clear all data for fresh testing
        db.session.commit()

    app.run(debug=True)  # Start the Flask app