import streamlit as st
import numpy as np
import pandas as pd
import json
import io

# Import your custom functions
from psdtool_app_functions import (
    check_samples, check_stat, fitting_function, plot_sample, reprep_Dvalue,
    LOGNORMAL, GIG, WEIBULL
)

# Page configuration
st.set_page_config(
    page_title="PSD Fitting Tool",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Particle Size Distribution Fitting Tool")
st.markdown("---")

# Initialize session state
if 'func_results' not in st.session_state:
    st.session_state.func_results = {}
if 'data_submitted' not in st.session_state:
    st.session_state.data_submitted = False
if 'y' not in st.session_state:
    st.session_state.y = None
if 'D' not in st.session_state:
    st.session_state.D = None

# Step 1: Data Input
st.header("Step 1: Input Data")
col1, col2 = st.columns(2)

with col1:
    sizes_input = st.text_input(
        "Sample Cumulative Distribution Grain sizes (comma-separated)",
        placeholder="Example: 0.25, 0.5, 1, 2, 4, 8",
        help="Enter particle sizes separated by commas"
    )

with col2:
    percentiles_input = st.text_input(
        "Sample Cumulatuive Distribution percentiles (comma-separated)",
        placeholder="Example: 1, 20, 40, 80",
        help="Enter percentile values separated by commas"
    )

if st.button("Submit Data", type="primary"):
    try:
        # Parse sizes (split by comma)
        sizes = [float(i.strip()) for i in sizes_input.split(',')]
        percentile = [float(i.strip()) for i in percentiles_input.split(',')]
        
        # Validation
        if len(sizes) == 0:
            st.error("⚠️ No size values entered")
        elif len(percentile) == 0:
            st.error("⚠️ No percentile values entered")
        elif len(sizes) != len(percentile):
            st.error(f"⚠️ Length mismatch! Sizes: {len(sizes)}, Percentiles: {len(percentile)}")
        else:
            # Create dataframe
            data = pd.DataFrame(columns=[str(i) for i in sizes])
            data.loc[0, data.columns] = percentile
            y = data.iloc[0].dropna()
            D = sizes.copy()
            
            # Store in session state
            st.session_state.y = y
            st.session_state.D = D
            st.session_state.data_submitted = True
            st.session_state.func_results = {}  # Reset results

            # Reset checkbox states when new data is submitted
            if 'show_plot' in st.session_state:
                st.session_state['show_plot'] = False
            
            st.success("✅ Data submitted successfully!")
            
    except ValueError:
        st.error("⚠️ Invalid input - please use only numbers separated by commas")

# Display submitted data
if st.session_state.data_submitted:
    st.markdown("---")
    st.subheader("Your Sample PSD is:")
    
    display_data = pd.DataFrame({
        'Grain Size': st.session_state.D,
        'Cumulative Percentile': st.session_state.y.values
    })
    st.table(display_data.T)#, width='stretch')#use_container_width=True)

    # Check samples - display as regular output
    import sys
    from io import StringIO
    
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    check_samples(st.session_state.y, st.session_state.D)
    
    # Get the captured output
    sys.stdout = old_stdout
    output = captured_output.getvalue()

    if output.strip():
        lines = output.strip().split('\n')
        for line in lines:
            if 'WARNING' in line or 'ERROR' in line:
                st.error(line)
            elif 'Success' in line or 'satisfies' in line:
                st.success(line)
            else:
                st.info(line)
    
# Step 2.1: Statistics Check
    st.markdown("---")
    st.header("Step 2: Fit Functions to the Sample PSD")
    
    available_functions = {
        'Lognormal': LOGNORMAL,
        'Weibull': WEIBULL,
        'GIG': GIG
    }
    st.info(f"Available functions: {', '.join(available_functions.keys())}") # Display available functions
    st.markdown(r"""The Sample PSD statistics: """)
    stats = check_stat(st.session_state.y, st.session_state.D)
    st.session_state.stats = stats

    if stats:
        # Display as formatted table like fitted results
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']
        st.table(stats_df.T)#, width='stretch') #use_container_width=True)
    
# Step 2.2: Function Fitting

    if st.button("Fit Functions"):
    
        with st.spinner(f"Fitting function..."):
            results = fitting_function(GIG, st.session_state.y, st.session_state.D)
            st.session_state.func_results['GIG'] = results

            results = fitting_function(LOGNORMAL, st.session_state.y, st.session_state.D)
            st.session_state.func_results['Lognormal'] = results

            results = fitting_function(WEIBULL, st.session_state.y, st.session_state.D)
            st.session_state.func_results['Weibull'] = results

        st.success(f"✅ Functions fitted successfully!")

        with st.spinner(f"Calculating D16, D50 and D84..."):
            for func_name in st.session_state.func_results.keys():
                # Calculate D16, D50, D84 in a loop
                for dv_name, dv_value in [('D16', 16), ('D50', 50), ('D84', 84)]:
                    d_result, _ = reprep_Dvalue(
                        y=st.session_state.y, 
                        D=st.session_state.D, 
                        DV=dv_value, 
                        Function=available_functions[func_name], 
                        A=st.session_state.func_results[func_name]['fitted_A'], 
                        B=st.session_state.func_results[func_name]['fitted_B'], 
                        numpoints=10000000
                    )
                    st.session_state.func_results[func_name][dv_name] = d_result
        
        st.success(f"✅ D16, D50, D84 calculated successfully!")

    # Display all fitted functions
    if st.session_state.func_results:
        st.header("Summary of Fitted Functions")
        st.markdown(r"""
                    **Notes:**
                    - For Lognormal: parametr A - is $\sigma$ and parameter B - is $\mu$ (parameters in Wikipedia page)
                    - For Weibull: parametr A - is $\lambda$ and parameter B - is k (parameters in Wikipedia page)
                    - For GIG: parameter A - is $\lambda$ in theoretical form, parameter B - is $\beta$
                    """)
        
        summary_data = []
        for func_name, result_dict in st.session_state.func_results.items():
            row = {'Function': func_name.upper()}
            row.update(result_dict)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)#, width='stretch') #use_container_width=True)
            
        combined_results = {}

        # Add statistics 
        if 'stats' in st.session_state and st.session_state.stats:
            for key, value in st.session_state.stats.items():
                combined_results[f'stats_{key}'] = value

        # Add fitted function results with function names in column headers
        for func_name, results_dict in st.session_state.func_results.items():
            for key, value in results_dict.items():
                # Create column name like "RMSE_lognormal" or "fitted_A_Weibull"
                column_name = f'{key}_{func_name}'
                combined_results[column_name] = value

        # Convert to DataFrame (single row)
        combined_df = pd.DataFrame([combined_results])

        # Download results as .CSV file
        csv = combined_df.to_csv(index=False)
        save_csv_name = st.text_input(
            "Enter Station ID and Sample Number to save output results as CSV file",
            placeholder="Station ID and Sample #01"
        )
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"{save_csv_name}.csv",
            mime="text/csv"
        )
        
# Step 3: Plotting
        st.markdown("---")
        st.header("Step 3: Visualization (Optional)")
        st.markdown(r"""
            **Note:**
            - The Cumulative Distribution of data is the plot of the provided sample.
            - The Cumulative Curve of the Function is the resulting fitted function plot.
            - The Probablity Density bar plot of data is calculated as the difference between two consecutive points in provided sample. 
            - The Probablity Density curve of the Function is calculated as the derivative of the fitted function: 
                    the difference between the points in a distribution divided by the step (distance between the points). 
            """)
        
        #if st.checkbox("Show plot of all fitted functions"):
        if st.checkbox("Show plot of all fitted functions", key="show_plot"):
            with st.spinner("Generating plot..."):
                fig = plot_sample(
                    st.session_state.y, 
                    st.session_state.D, 
                    st.session_state.func_results, 
                    save_as=False
                )
                st.pyplot(fig)
            
            # Download the plot as .PNG file
            save_plot_name = st.text_input(
                "Enter filename to save plot (optional)",
                placeholder="my_plot"
            )
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=450)
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Plot (PNG)",
                data=buf,
                file_name=f"{save_plot_name}.png",
                mime="image/png"
            )

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    ### How to use:
    
    1. **Input Data**: Enter sample cumulative distribution sizes and percentiles (comma-separated)
    2. **Check Statistics**: View sample PSD statistics
    3. **Fit Functions**:Fit available functions
    4. **View Results**: See Functon's fitted metrics, parameters and representative D values in a summary table
    5. **Download**: Export combined output table as CSV file
    6. **Visualize**: Plot all fitted functions together
    7. **Download**: Export plots (optionally)
    
    """)
    
    if st.session_state.func_results:
        st.markdown("---")
        st.metric("Functions Fitted", len(st.session_state.func_results))
    
    # Reset button
    if st.button("🔄 Reset All"):
        st.session_state.func_results = {}
        st.session_state.data_submitted = False
        st.session_state.y = None
        st.session_state.D = None
        st.rerun()