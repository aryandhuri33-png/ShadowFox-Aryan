from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import datetime

# ✅ Fix: Use double underscores
app = Flask(__name__)

# ✅ Updated: Load the trained model
model = joblib.load("car_price_model.pkl")

# ✅ Keep your HTML unchanged
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CarWise PRO - AI Car Price Predictor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');
        body {
            margin: 0;
            font-family: 'Rubik', sans-serif;
            background: linear-gradient(to right, #e0f7fa, #c8e6c9);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            width: 95%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            text-align: center;
            color: #2e7d32;
        }
        form {
            width: 100%;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group label {
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
        }
        .form-group input,
        .form-group select {
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        button {
            width: 100%;
            margin-top: 20px;
            padding: 14px;
            font-size: 1.1rem;
            background: linear-gradient(135deg, #43cea2, #185a9d);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.3s ease;
        }
        button:hover {
            transform: scale(1.03);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .result-box {
            margin-top: 2rem;
            display: flex;
            justify-content: center;
            animation: fadeIn 1s ease-in-out;
        }
        .price-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 2rem 3rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.2);
            animation: slideIn 1s ease forwards;
            color: #1b5e20;
        }
        .price-title {
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #2e7d32;
        }
        .price-value {
            font-size: 3.2rem;
            font-weight: 700;
            color: #388e3c;
            text-shadow: 0 0 10px rgba(56, 142, 60, 0.3);
            transition: all 0.4s ease-in-out;
        }
        .price-sub {
            font-size: 1rem;
            color: #4caf50;
            margin-top: 0.2rem;
        }
        .plot-container {
            margin-top: 2rem;
            text-align: center;
            animation: fadeSlide 0.9s ease-out forwards;
            opacity: 0;
        }
        @keyframes slideIn {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeSlide {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0px); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 CarWise PRO</h1>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label>Showroom Price (in Lakhs)</label>
                    <input type="number" name="present_price" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Kilometers Driven</label>
                    <input type="number" name="kms_driven" required>
                </div>
                <div class="form-group">
                    <label>Previous Owners</label>
                    <input type="number" name="owner" min="0" max="3" required>
                </div>
                <div class="form-group">
                    <label>Fuel Type</label>
                    <select name="fuel_type">
                        <option value="Petrol">Petrol</option>
                        <option value="Diesel">Diesel</option>
                        <option value="CNG">CNG</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Seller Type</label>
                    <select name="seller_type">
                        <option value="Dealer">Dealer</option>
                        <option value="Individual">Individual</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Transmission Type</label>
                    <select name="transmission">
                        <option value="Manual">Manual</option>
                        <option value="Automatic">Automatic</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Year of Purchase</label>
                    <input type="number" name="year" min="2000" max="2025" required>
                </div>
            </div>
            <button type="submit">🔍 Predict Selling Price</button>
        </form>

        {% if predicted_price is not none %}
        <div class="result-box">
            <div class="price-card">
                <div class="price-title">💰 Estimated Selling Price</div>
                <div class="price-value" id="countUpPrice">0</div>
                <div class="price-sub">in Lakhs</div>
            </div>
        </div>
        <script>
            const targetPrice = {{ predicted_price }};
            const duration = 1500;
            const fps = 60;
            const totalFrames = Math.round(duration / (1000 / fps));
            let frame = 0;
            const counter = setInterval(() => {
                frame++;
                const progress = frame / totalFrames;
                const currentValue = (targetPrice * progress).toFixed(2);
                document.getElementById("countUpPrice").textContent = currentValue;
                if (frame === totalFrames) clearInterval(counter);
            }, 1000 / fps);
        </script>
        {% endif %}

        {% if plot_data %}
        <div class="plot-container">
            <h3>📊 Showroom vs Predicted Price</h3>
            <img src="data:image/png;base64,{{ plot_data }}" alt="Prediction Chart">
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    plot_data = None

    if request.method == 'POST':
        try:
            present_price = float(request.form['present_price'])
            kms_driven = int(request.form['kms_driven'])
            owner = int(request.form['owner'])
            fuel_type = request.form['fuel_type']
            seller_type = request.form['seller_type']
            transmission = request.form['transmission']
            purchase_year = int(request.form['year'])
            car_age = datetime.datetime.now().year - purchase_year

            input_df = pd.DataFrame([{
                'Present_Price': present_price,
                'Kms_Driven': kms_driven,
                'Owner': owner,
                'Fuel_Type': fuel_type,
                'Seller_Type': seller_type,
                'Transmission': transmission,
                'car_age': car_age
            }])

            predicted_price = round(model.predict(input_df)[0], 2)

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
            predicted_price = f"Error: {e}"

    return render_template_string(HTML, predicted_price=predicted_price, plot_data=plot_data)

# ✅ Fix: Double underscores
if __name__ == '__main__':
    app.run(debug=True)
