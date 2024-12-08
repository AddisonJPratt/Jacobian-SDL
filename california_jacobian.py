
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(page_title="California Housing NN + Advanced Jacobian Analysis", layout="wide")

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
@st.cache_data
def load_and_preprocess_data():
    california = fetch_california_housing()
    X, y = california.data, california.target

    # Multi-output: original and log-transformed target
    y_log = np.log(y + 1)
    y_multi = np.vstack((y, y_log)).T

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = california.feature_names
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X_train, X_test, y, california

# ------------------------------
# Neural Network Model
# ------------------------------
class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiOutputNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

def train_model(X_train, y_train, hidden_dim=64, learning_rate=0.001, num_epochs=1000):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MultiOutputNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        criterion = nn.MSELoss()
        test_loss = criterion(predictions, y_test_tensor).item()
    return predictions.numpy(), test_loss

def compute_jacobian(model, sample_input):
    def model_forward(x):
        return model(x)
    return jacobian(model_forward, sample_input)

def global_sensitivity_approx(model, X_test_tensor, n_samples=100):
    # Randomly select n_samples from test set to compute average sensitivity
    indices = np.random.choice(X_test_tensor.shape[0], n_samples, replace=False)
    jacobians = []
    for idx in indices:
        inp = X_test_tensor[idx].unsqueeze(0).requires_grad_(True)
        J = compute_jacobian(model, inp).detach().numpy().squeeze()
        jacobians.append(J)
    avg_jac = np.mean(jacobians, axis=0)
    return avg_jac

# --------------
# Load Data
# --------------
X_train, X_test, y_train, y_test, feature_names, X_train_raw, X_test_raw, y_raw, california = load_and_preprocess_data()
output_names = ['MedHouseVal', 'Log(MedHouseVal)']

st.sidebar.header("Model Settings")
hidden_dim = st.sidebar.slider("Hidden Layer Size", min_value=32, max_value=256, step=32, value=64)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)
num_epochs = st.sidebar.slider("Training Epochs", min_value=500, max_value=5000, step=500, value=1500)

st.sidebar.header("Jacobian Analysis")
sample_index = st.sidebar.slider("Test Sample Index", min_value=0, max_value=X_test.shape[0]-1, step=1, value=0)
global_sample_count = st.sidebar.slider("Samples for Global Sensitivity", min_value=10, max_value=300, step=10, value=50)

# Train the model
model = train_model(X_train, y_train, hidden_dim=hidden_dim, learning_rate=learning_rate, num_epochs=num_epochs)
predictions, test_loss = evaluate_model(model, X_test, y_test)

tab1, tab2, tab3, tab4 = st.tabs(["Data & Model", "Evaluation", "Jacobian Analysis", "Additional Insights"])

with tab1:
    st.title("Data & Model")
    st.markdown("### California Housing Dataset Overview")
    st.write(f"**Features:** {feature_names}")
    st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    df_all = pd.DataFrame(X_train_raw, columns=feature_names)
    corr = df_all.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix",
                         aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("### Model Architecture")
    st.write(f"Hidden Dimension: {hidden_dim}")
    st.write(f"Learning Rate: {learning_rate}")
    st.write(f"Epochs: {num_epochs}")
    st.markdown("""
    **Model:**
    - Input: 8 features  
    - 2 Hidden Layers (ReLU activation)  
    - Output: 2 (Median House Value & Log(Median House Value))
    """)

with tab2:
    st.title("Evaluation")
    st.write(f"**Test MSE Loss:** {test_loss:.4f}")

    actual_values = y_test[:,0]
    predicted_values = predictions[:,0]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=actual_values, y=predicted_values, mode='markers', 
                                     marker=dict(color='blue', size=6),
                                     name='Predictions'))
    
    chosen_sample_actual = actual_values[sample_index]
    chosen_sample_pred = predicted_values[sample_index]
    fig_scatter.add_trace(go.Scatter(x=[chosen_sample_actual], y=[chosen_sample_pred], mode='markers',
                                     marker=dict(color='red', size=10, symbol='x'), 
                                     name='Chosen Sample'))
    
    fig_scatter.update_layout(title="Predicted vs Actual Median House Values",
                              xaxis_title="Actual Values",
                              yaxis_title="Predicted Values",
                              template='plotly_white')

    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.title("Jacobian Analysis")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    sample_input = X_test_tensor[sample_index].unsqueeze(0).requires_grad_(True)
    jacobian_matrix = compute_jacobian(model, sample_input)
    jacobian_np = jacobian_matrix.detach().numpy().squeeze()

    st.write(f"**Analyzing Sample Index:** {sample_index}")
    st.write("Features (Scaled):", X_test[sample_index])

    df_jacobian = pd.DataFrame(jacobian_np, index=output_names, columns=feature_names)
    st.write("**Local Jacobian Matrix:**")
    st.dataframe(df_jacobian.style.background_gradient(cmap='RdBu', axis=1))

    fig_local_heat = px.imshow(df_jacobian, x=feature_names, y=output_names,
                               title="Jacobian Heatmap (Local Sensitivity)",
                               color_continuous_scale='RdBu', aspect='auto')
    st.plotly_chart(fig_local_heat, use_container_width=True)

    abs_jacobian = np.abs(jacobian_np)
    for i, out_name in enumerate(output_names):
        abs_sens_df = pd.DataFrame({
            'Feature': feature_names,
            'Absolute Sensitivity': abs_jacobian[i]
        }).sort_values('Absolute Sensitivity', ascending=False)
        
        fig_bar = px.bar(abs_sens_df, x='Feature', y='Absolute Sensitivity',
                         title=f'Absolute Sensitivity for {out_name}',
                         color='Absolute Sensitivity', 
                         color_continuous_scale='Plasma')
        fig_bar.update_layout(template="plotly_white", xaxis_title="Features", yaxis_title="|Jacobian|")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Global Sensitivity Approximation")
    avg_jac = global_sensitivity_approx(model, X_test_tensor, n_samples=global_sample_count)
    avg_abs_jac = np.abs(avg_jac)
    global_sens_df = pd.DataFrame(avg_abs_jac, index=output_names, columns=feature_names)
    st.markdown(f"**Averaged over {global_sample_count} random test samples**:")
    st.dataframe(global_sens_df.style.background_gradient(cmap='Blues'))

    # Additional Interpretation Section
    st.markdown("### Interpreting the Jacobian Values")

    st.markdown("""
    Each value in the Jacobian matrix represents how much a small change in a given feature would affect the model's prediction for a particular output **at this specific sample**.

    - **Positive Value:**  
      Increasing this feature slightly (holding all others constant) **increases** the predicted output.
    
    - **Negative Value:**  
      Increasing this feature slightly **decreases** the predicted output.
      
    The magnitude indicates how sensitive the prediction is to small changes in that feature. Larger magnitudes mean higher sensitivity.
    """)

    # Highlighting interesting values (top sensitivities)
    st.markdown("**Key Observations for This Sample:**")
    for i, out_name in enumerate(output_names):
        sorted_sens = pd.DataFrame({
            'Feature': feature_names,
            'Absolute Sensitivity': abs_jacobian[i]
        }).sort_values('Absolute Sensitivity', ascending=False)
        top_feature = sorted_sens.iloc[0]['Feature']
        top_value = df_jacobian.loc[out_name, top_feature]

        second_feature = sorted_sens.iloc[1]['Feature']
        second_value = df_jacobian.loc[out_name, second_feature]

        st.markdown(f"For output **{out_name}**:")
        st.write(f"- The most influential feature is **{top_feature}** with a Jacobian of {top_value:.4f}.")
        if top_value > 0:
            st.write(f"  Increasing {top_feature} slightly tends to **increase** the predicted {out_name}.")
        else:
            st.write(f"  Increasing {top_feature} slightly tends to **decrease** the predicted {out_name}.")

        st.write(f"- The second most influential feature is **{second_feature}** with a Jacobian of {second_value:.4f}.")
        if second_value > 0:
            st.write(f"  Increasing {second_feature} slightly also tends to **increase** the predicted {out_name}.")
        else:
            st.write(f"  Increasing {second_feature} slightly tends to **decrease** the predicted {out_name}.")
        st.write(" ")

    st.markdown("""
    ### Local vs. Global Sensitivities

    - **Local Sensitivity (Jacobian at One Point):**  
      The values are local to the chosen sample. A positive Jacobian for latitude, for example, means *at that sample's location*, moving slightly north could increase predicted prices. It's not a universal rule.

    - **Global Sensitivity (Averaged Across Many Samples):**  
      Averaging across samples gives a general sense of which features are influential overall.

    These perspectives complement each other. Local analysis reveals instance-level interpretations, while global analysis guides overall feature importance understanding.
    """)

    st.markdown("### Interpreting Latitude & Longitude Influences")

    st.markdown("""
    The dataset includes **latitude** and **longitude**, which serve as proxies for location. The model may learn patterns such as:
    
    - Homes in certain northern latitudes (like the Bay Area region) are more expensive due to strong job markets, limited housing supply, and desirable amenities.
    - A positive Jacobian w.r.t. latitude at your sample suggests that, in that neighborhood, moving slightly north aligns with higher-value areas. This is a local interpretation and doesn't mean latitude universally increases home prices everywhere.
    
    Underlying factors could be:
    - Proximity to high-paying jobs.
    - Climate or environmental preferences.
    - Historical development patterns.
    """)

    with st.expander("Mathematical Details"):
        st.markdown(r"""
        Consider the model as a function $ f: \mathbb{R}^n \to \mathbb{R}^m $. In our case:
        
        - Inputs: $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, where some of these features include **latitude** and **longitude**.
        - Outputs: $\mathbf{y} = f(\mathbf{x}) = (y_1, y_2)$, representing median house value and its log-transform.
        
        The **Jacobian** $J$ at a point $\mathbf{x}$ is:
        $$
        J(\mathbf{x}) = \begin{bmatrix}
        \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\
        \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n}
        \end{bmatrix}
        $$
        
        If the feature $x_j$ corresponds to latitude, then $\frac{\partial y_i}{\partial \text{latitude}}$ tells us how the \(i\)-th output changes with a tiny increase in latitude, holding other features constant.

        **Local Interpretation:**  
        For a specific sample $\mathbf{x}_0$:
        $$
        \frac{\partial y_i}{\partial \text{latitude}}\bigg|_{\mathbf{x}=\mathbf{x}_0} > 0 \implies \text{In the vicinity of }\mathbf{x}_0, \text{ increasing latitude slightly increases } y_i.
        $$

        This is only guaranteed near $\mathbf{x}_0$. Far from this point, the relationship may change.

        Thus, the Jacobian is a **first-order approximation** of how the outputs respond to input changes locally, not a universal mapping across the entire input space.
        """)

with tab4:
    st.title("Additional Insights")

    st.markdown("### Feature Distributions")
    df_all_test = pd.DataFrame(X_test, columns=feature_names)
    chosen_sample_features = X_test[sample_index]

    for i, feat in enumerate(feature_names):
        fig_hist = px.histogram(df_all_test, x=feat, nbins=50, title=f"Distribution of {feat}")
        fig_hist.add_vline(x=chosen_sample_features[i], line_width=3, line_dash="dash", line_color="red")
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(r"""
    ### Local Linear Approximation

    The Jacobian provides the gradient of the model at a point $\mathbf{x}$:
    $$
    f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + J(\mathbf{x}) \Delta \mathbf{x}
    $$

    This lets us estimate how small changes in features would shift the predictions at that specific point.
    """)

    st.markdown("""
    ### Comparisons with Other Interpretability Methods

    - **SHAP / LIME:** Use perturbations and surrogate models to estimate feature importance.  
    - **Integrated Gradients:** Averages gradients along a path from a baseline.  
    - **Jacobian:** Direct local gradient at a single point.

    By combining these methods, you gain a more comprehensive understanding of your model.
    """)

st.markdown("""
---
**Summary:**  
- We trained a multi-output neural network on the California Housing dataset.
- Performed local sensitivity analysis using Jacobians to understand how each feature affects predictions for a specific instance.
- Examined global sensitivities by averaging Jacobians across many samples to find stable, overall patterns.
- Interpreted the meaning of latitude and longitude sensitivities, and provided mathematical and conceptual frameworks for understanding the Jacobian.
""")
# %%