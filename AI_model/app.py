
from flask import Flask, render_template, request, redirect, url_for, abort
from data import klanten, producten, aankoopgeschiedenis
from recommend import find_combo_deals, get_product_by_id
from sklearn_recommend import (
    get_all_sklearn_recommendations,
    get_sklearn_recommendations,
    train_sklearn_model,
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user')
        if user_id and user_id in [k['id'] for k in klanten]:
            return redirect(url_for('dashboard', user_id=user_id))
    return render_template('login.html', klanten=klanten)

@app.route('/dashboard/<user_id>', methods=['GET', 'POST'])
def dashboard(user_id):
    klant = next((k for k in klanten if k['id'] == user_id), None)
    if not klant:
        return redirect(url_for('login'))
    aankoop = aankoopgeschiedenis.get(user_id, [])

    # Sklearn aanbevelingen
    try:
        sklearn_recommendations = get_sklearn_recommendations(user_id, num_recommendations=5)
    except Exception as e:
        print(f"Sklearn error: {e}")
        sklearn_recommendations = []

    # Combo deals
    combo_deals = find_combo_deals(user_id)
    # find_combo_deals returns (product1_dict, product2_dict, score) tuples already
    deals_detail = combo_deals

    return render_template('dashboard.html', 
                         klant=klant, 
                         aankoop=aankoop, 
                         deals=deals_detail, 
                         sklearn_recommendations=sklearn_recommendations,
                         producten=producten, 
                         get_product_by_id=get_product_by_id)


@app.route('/recommendations')
def all_recommendations():
    all_recs = get_all_sklearn_recommendations(num_recommendations=3)

    # Verrijk met klantdata voor weergave
    klant_by_id = {k['id']: k for k in klanten}
    enriched = []
    for user_id, recommendations in all_recs.items():
        klant = klant_by_id.get(user_id)
        enriched.append((klant, recommendations))

    enriched.sort(key=lambda item: (item[0]['naam'] if item[0] else '',))

    return render_template(
        'recommendations.html',
        aanbevelingen=enriched,
    )

@app.route('/train-model')
def train_model():
    """Train the sklearn model"""
    try:
        train_sklearn_model()
        return "Model trained successfully!"
    except Exception as e:
        return f"Error training model: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
