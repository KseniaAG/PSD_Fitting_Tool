# Particle Size Distribution Fitting Tool

A web application for fitting particle size distribution (PSD) data to different mathematical functions. This tool helps you by fitting a sample PSD to the Lognormal, Weibull, and Generalized Inverse Gaussian distributions, providing estimated parameters for each function and showing how well each function matches your data via metrics and plots. 

# This App is available via the link: 
https://psd-fittingtool-kaguh.streamlit.app

## How to Use the App

### Step 1: Enter Your Data
- Enter the particle sizes from the sample PSD separated by commas (example: 0.25, 0.5, 1, 2, 4, 8)
- Enter the cumulative percentile values  from the sample PSD separated by commas (example: 1, 20, 40, 80, 100)
- Click "Submit Data"
- Check the submitted sample PSD for representativeness

### Step 2: Fit Functions to Sample PSD
- You can view statistical information of the provided sample PSD
- Click "Fit Selected Function"
- See Functon's fitted metrics, parameters and representative D values in a summary table
- Export combined output table as CSV file

### Step 3: View Results
- View plots showing your data and fitted curves
- Download the plots (optional)

## Requirements

This app uses the following Python packages:
- streamlit - for the web interface
- numpy 
- pandas
- matplotlib
- scipy
- scikit-learn

## To run the App Locally, you need to: 

If you want to run this app on your own computer:

1. Install Python (version 3.8 or higher)
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the app from the Terminal(VSCode or iOS)/Command Line(Windows):
   ```
   streamlit run app.py
   ```
4. The App will automatically open in your web browser

## Files in This Project
- `psdtool_app.py` - The main application file
- `psdtool_app_functions.py` - Contains the calculation and plotting functions
- `requirements.txt` - List of required Python packages
- `README.md` - This file

## Privacy Notice

This application processes data in your browser session. Your data is not stored permanently and is cleared when you close the browser or click the "Reset" button at the bottom of the sidebar.

## Support

If you encounter any issues or have questions, please check that:
- All required packages are installed
- Your input sample PSD is he cumulative particle size distribution data and it is formatted correctly
- The number of size values matches the number of percentile values
- For any other issues, please contact to author. 

## License

This project is for research and educational purposes.
