from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import folium
import io
import base64

app = Flask(__name__)
model_filename = 'property_price_model.pkl'
loaded_model = joblib.load(model_filename)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    street = request.form['street']
    area = float(request.form['area'])
    year_built = int(request.form['year_built'])
    property_type = request.form['property_type']
    nearby_locations = request.form['nearby_locations']
    crime_rate = float(request.form['crime_rate'])
    water_facility = request.form.get('water_facility')
    public_transport = request.form['public_transport']
    
    # Create DataFrame
    user_input_df = pd.DataFrame({
        'Location': [location],
        'Street': [street],
        'Area (sq ft)': [area],
        'Year Built': [year_built],
        'Property Type': [property_type],
        'Nearby Famous Locations': [nearby_locations],
        'Crime Rate': [crime_rate],
        'Public Transport Accessibility': [public_transport],
        'Water Facility': [water_facility] if property_type != 'Plot' else [None]
    })

    # Predict price
    predicted_price = loaded_model.predict(user_input_df)[0]

    # Create plot
    average_areas = {
        'Porur': 1200, 'Anna Nagar': 1500, 'T. Nagar': 1600, 'Adyar': 1400,
        'Velachery': 1300, 'Guindy': 1350, 'Mylapore': 1450, 'Kodambakkam': 1250,
        'Nungambakkam': 1550, 'Tambaram': 1100
    }
    
    fig, ax = plt.subplots()
    ax.barh(list(average_areas.keys()), list(average_areas.values()), color="#1E90FF", alpha=0.6)
    ax.barh([location], [area], color="orange", alpha=0.8)
    ax.set_xlabel("Area (sq ft)")
    ax.set_title("Your Area vs. Average Area in Locations")

    # Save plot to a BytesIO object and encode it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # Create map
    location_coords = {
        'Porur': [13.0358, 80.1588],
        'Anna Nagar': [13.0878, 80.2083],
        'T. Nagar': [13.0409, 80.2333],
        'Adyar': [13.0057, 80.2574],
        'Velachery': [12.9791, 80.2209],
        'Guindy': [13.0103, 80.2125],
        'Mylapore': [13.0305, 80.2665],
        'Kodambakkam': [13.0564, 80.2214],
        'Nungambakkam': [13.0606, 80.2478],
        'Tambaram': [12.9242, 80.1278]
    }
    
    map_location = folium.Map(location=location_coords[location], zoom_start=13)
    folium.Marker(location_coords[location], tooltip=location).add_to(map_location)
    
    # Save map to an HTML file
    map_html_path = 'static/map.html'
    map_location.save(map_html_path)

    return render_template('result.html', 
                           predicted_price=f"{predicted_price:,.2f}", 
                           user_input_df=user_input_df.to_html(), 
                           plot_data=plot_data,
                           map_html_path=map_html_path)

if __name__ == '__main__':
    app.run(debug=True)
