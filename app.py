# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression, Lasso, Ridge
# # from sklearn.metrics import mean_squared_error, r2_score

# # # Create dropdown for regression model selection
# # model_options = {
# #     "Linear Regression": LinearRegression,
# #     "Lasso Regression": Lasso,
# #     "Ridge Regression": Ridge
# # }

# # # Title of the app
# # st.title('Regression Model Comparison App')

# # # Upload the dataset
# # uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# # # Load dataset
# # if uploaded_file is not None:
# #     df = pd.read_csv(uploaded_file)
# #     st.write("Data Preview:")
# #     st.write(df.head())

# #     # Input feature selection (X) and target variable (y)
# #     X = st.multiselect('Select Features (X)', df.columns)
# #     y = st.selectbox('Select Target (y)', df.columns)

# #     if X and y:
# #         # Test size slider
# #         test_size = st.slider('Test Size (0.1 - 0.5)', min_value=0.1, max_value=0.5, step=0.01, value=0.2)

# #         # Random state input
# #         random_state = st.number_input('Random State', min_value=0, value=42)

# #         # Model selection dropdown
# #         selected_model = st.selectbox('Select Regression Model', list(model_options.keys()))

# #         # Train button
# #         if st.button("Train Model"):
# #             # Splitting the data
# #             X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=test_size, random_state=random_state)

# #             # Model initialization
# #             model = model_options[selected_model]()  # Initialize the selected model
# #             model.fit(X_train, y_train)

# #             # Make predictions
# #             y_pred = model.predict(X_test)

# #             # Calculate metrics
# #             mse = mean_squared_error(y_test, y_pred)
# #             r2 = r2_score(y_test, y_pred)

# #             # Display results
# #             st.write(f"### {selected_model} Results:")
# #             st.write(f"Mean Squared Error: {mse}")
# #             st.write(f"R-squared: {r2}")

# #             # Optionally, show a scatter plot of predictions vs actual values
# #             st.subheader('Prediction vs Actual Plot')
# #             plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# #             st.line_chart(plot_data)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Create dropdown for regression model selection
model_options = {
    "Linear Regression": LinearRegression,
    "Lasso Regression": Lasso,
    "Ridge Regression": Ridge
}

# Title of the app
st.title('Regression Model Comparison App')

# Upload the dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Input feature selection (X) and target variable (y)
    X = st.multiselect('Select Features (X)', df.columns)
    y = st.selectbox('Select Target (y)', df.columns)

    if X and y:
        # Test size slider
        test_size = st.slider('Test Size (0.1 - 0.5)', min_value=0.1, max_value=0.5, step=0.01, value=0.2)

        # Random state input
        random_state = st.number_input('Random State', min_value=0, value=42)

        # Model selection dropdown
        selected_model = st.selectbox('Select Regression Model', list(model_options.keys()))

        # Train button
        if st.button("Train Model"):
            # Splitting the data
            X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=test_size, random_state=random_state)

            # Model initialization
            model = model_options[selected_model]()  # Initialize the selected model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.write(f"### {selected_model} Results:")
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R-squared (Accuracy): {r2}")  # Accuracy displayed as R-squared

            # Optionally, show a scatter plot of predictions vs actual values
            st.subheader('Prediction vs Actual Plot')
            plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.line_chart(plot_data)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.metrics import mean_squared_error, r2_score

# # Create dropdown for regression model selection
# model_options = {
#     "Linear Regression": LinearRegression,
#     "Lasso Regression": Lasso,
#     "Ridge Regression": Ridge
# }

# # Title of the app
# st.title('Regression Model Comparison App')

# # Upload the dataset
# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# # Load dataset
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("Data Preview:")
#     st.write(df.head())

#     # Input feature selection (X) and target variable (y)
#     X = st.multiselect('Select Features (X)', df.columns)
#     y = st.selectbox('Select Target (y)', df.columns)

#     if X and y:
#         # Test size slider
#         test_size = st.slider('Test Size (0.1 - 0.5)', min_value=0.1, max_value=0.5, step=0.01, value=0.2)

#         # Random state input
#         random_state = st.number_input('Random State', min_value=0, value=42)

#         # Model selection dropdown
#         selected_model = st.selectbox('Select Regression Model', list(model_options.keys()))

#         # Train button
#         if st.button("Train Model"):
#             # Splitting the data
#             X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=test_size, random_state=random_state)

#             # Model initialization
#             model = model_options[selected_model]()  # Initialize the selected model
#             model.fit(X_train, y_train)

#             # Make predictions
#             y_pred = model.predict(X_test)

#             # Calculate metrics
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)

#             # Display results
#             st.write(f"### {selected_model} Results:")
#             st.write(f"Mean Squared Error: {mse}")
#             st.write(f"R-squared (Accuracy): {r2}")  # Accuracy displayed as R-squared

#             # Display top 5 predicted vs actual values
#             st.write("### Top 5 Predicted vs Actual Values:")
#             results_df = pd.DataFrame({
#                 'Actual': y_test[:5].values,
#                 'Predicted': y_pred[:5]
#             })
#             st.write(results_df)

