import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output

# =============================
# Flights Dataset (Dummy Example)
# =============================
years = pd.date_range("1990", periods=36, freq="YE")
flight_data = {
    "date": years,
    "actual_revenue": np.linspace(50, 200, 36) + np.random.randn(36) * 10,
    "predicted_revenue": np.linspace(55, 210, 36) + np.random.randn(36) * 8,
    "passenger_return_rate": np.random.uniform(60, 90, 36),
}
flights = pd.DataFrame(flight_data)

# =============================
# Companies Dataset
# =============================
companies = pd.read_csv("Samples/Companies.csv")
companies["Revenue (Billions USD)"] = (
    companies["Revenue (Billions USD)"].astype(str).str.replace(",", "").astype(float)
)
companies["Employees"] = (
    companies["Employees"].astype(str).str.replace(",", "").replace("N/A", "0").astype(int)
)
companies["Market Cap (B)"] = (
    companies["Market Cap (B)"].astype(str).str.replace(",", "").replace("N/A", "0").astype(float)
)

# =============================
# Dash App
# =============================
app = dash.Dash(__name__)
app.title = "Unified Dashboard"

# Metric options for both datasets
flight_metrics = {
    "Revenue (Actual vs Predicted)": "revenue",
    "Passenger Return Rate": "return_rate",
}

company_metrics = {
    "Top 10 Companies by Revenue": "top10",
    "Revenue vs Market Cap Scatterplot": "scatter",
    "Distribution of Employees": "hist",
    "Correlation Heatmap": "heatmap",
    "Top 10 Countries by Revenue": "country_revenue",
}

# =============================
# Layout
# =============================
app.layout = html.Div([
    html.H1("✈️ Aviation & Business Insights Dashboard 📊", style={"textAlign": "center"}),

    html.Label("Select Dataset:", style={"fontWeight": "bold"}),
    dcc.Dropdown(
        id="dataset-dropdown",
        options=[{"label": "Flights", "value": "flights"},
                 {"label": "Companies", "value": "companies"}],
        value="flights",
        clearable=False,
        style={"width": "40%"}
    ),

    html.Label("Select Metric:", style={"fontWeight": "bold", "marginTop": "10px"}),
    dcc.Dropdown(id="metric-dropdown", clearable=False, style={"width": "60%"}),

    dcc.Graph(id="main-graph", style={"height": "600px"})
])


# =============================
# Callbacks
# =============================

# Update metric options dynamically
@app.callback(
    Output("metric-dropdown", "options"),
    Output("metric-dropdown", "value"),
    Input("dataset-dropdown", "value")
)
def update_metric_options(selected_dataset):
    if selected_dataset == "flights":
        opts = [{"label": k, "value": v} for k, v in flight_metrics.items()]
        return opts, "revenue"
    else:
        opts = [{"label": k, "value": v} for k, v in company_metrics.items()]
        return opts, "top10"


# Update graph based on dataset + metric
@app.callback(
    Output("main-graph", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("metric-dropdown", "value")]
)
def update_graph(dataset, metric):
    if dataset == "flights":
        if metric == "revenue":
            fig = px.line(flights, x="date", y=["actual_revenue", "predicted_revenue"],
                          title="Actual vs Predicted Revenue")
        elif metric == "return_rate":
            fig = px.line(flights, x="date", y="passenger_return_rate",
                          title="Passenger Return Rate Over Time")
        else:
            fig = px.scatter(title="⚠️ No Flight Visualization Found")

    elif dataset == "companies":
        if metric == "top10":
            top10 = companies.nlargest(10, "Revenue (Billions USD)")
            fig = px.bar(top10, x="Company Name", y="Revenue (Billions USD)",
                         color="Industry", title="Top 10 Companies by Revenue")
            fig.update_layout(xaxis_tickangle=-45)

        elif metric == "scatter":
            fig = px.scatter(companies, x="Revenue (Billions USD)", y="Market Cap (B)",
                             size="Employees", color="Industry", hover_name="Company Name",
                             title="Revenue vs Market Cap (Bubble size = Employees)",
                             size_max=60)

        elif metric == "hist":
            fig = px.histogram(companies, x="Employees", nbins=30,
                               title="Distribution of Employees")
            fig.update_traces(marker_color="blue")

        elif metric == "heatmap":
            corr = companies[["Revenue (Billions USD)", "Employees", "Market Cap (B)"]].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap of Financial Metrics")

        elif metric == "country_revenue":
            country_revenue = companies.groupby("Country")["Revenue (Billions USD)"].sum().reset_index()
            top_countries = country_revenue.sort_values(by="Revenue (Billions USD)",
                                                        ascending=False).head(10)
            fig = px.bar(top_countries, x="Revenue (Billions USD)", y="Country",
                         orientation="h", title="Top 10 Countries by Total Revenue",
                         color="Revenue (Billions USD)", color_continuous_scale="magma")
        else:
            fig = px.scatter(title="⚠️ No Company Visualization Found")
    else:
        fig = px.scatter(title="⚠️ No dataset selected")

    fig.update_layout(xaxis_title="Year", yaxis_title="Value")
    return fig


# =============================
# Run
# =============================
if __name__ == "__main__":
    app.run(debug=True)

