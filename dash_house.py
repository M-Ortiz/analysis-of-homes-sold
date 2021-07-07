import pandas as pd
#import geopandas
import numpy as np
import plotly.express as px
import streamlit as st
import folium
from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster

st.set_page_config( layout="wide" )

st.title( "House Rockets Company" )
st.markdown( "Welcome to House Rockect Data Analysis" )

st.header( "Load Data" )

# Read Data
@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )

    return data


def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile


def set_feature( data ):
    data["price_m2"] = data["price"] / data["sqft_lot"]
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

    return data


def overview_data( data ):
    #  -------------------------------------------- Data Overview
    st.title("Data Overview")

    # Filter
    f_atribures = st.sidebar.multiselect("Enter columns", data.columns)
    f_zipcode = st.sidebar.multiselect("Enter zipcode", data['zipcode'].unique())

    if (f_zipcode != []) & (f_atribures != []):
        data = data.loc[data["zipcode"].isin(f_zipcode), f_atribures]

    elif (f_zipcode != []) & (f_atribures == []):
        data = data.loc[data["zipcode"].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_atribures != []):
        data = data.loc[:, f_atribures]
    else:
        data = data.copy()

    st.dataframe( data )

    # Layout Page
    c1, c2 = st.beta_columns((1, 1))

    #  -------------------------------------------- Average Metrics
    c1.header("Average Metrics")
    # Average metrics
    df1 = data[['id', 'zipcode']].groupby("zipcode").count().reset_index()
    df2 = data[["price", "zipcode"]].groupby("zipcode").mean().reset_index()
    df3 = data[["sqft_living", "zipcode"]].groupby("zipcode").mean().reset_index()
    df4 = data[["price_m2", "zipcode"]].groupby("zipcode").mean().reset_index()
    # merge data
    m1 = pd.merge(df1, df2, on="zipcode", how="inner")
    m2 = pd.merge(m1, df3, on="zipcode", how="inner")
    df = pd.merge(m2, df4, on="zipcode", how="inner")
    # Rename columns
    df.columns = ["ZIPCODE", "TOTAL HOUSES", "PRICE", "SQFT LIVING", "PRICE/M2"]
    # print the dataframe
    c1.dataframe(df)

    # --------------------------------------------  Statistic Descriptive
    nun_atributes = data.select_dtypes(include=["int64", "float64"])
    # Tranformation
    media = pd.DataFrame(nun_atributes.apply(np.mean))
    mediana = pd.DataFrame(nun_atributes.apply(np.median))
    std = pd.DataFrame(nun_atributes.apply(np.std))
    max_ = pd.DataFrame(nun_atributes.apply(np.max))
    min_ = pd.DataFrame(nun_atributes.apply(np.min))

    # concat
    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    # Rename columns
    df1.columns = ["ATTRIBUTES", "MAX", "MIN", "MEAN", "MEDIAN", "STD"]
    # Nome datafeame
    c2.header("Statistic Descriptive")
    # Print dataframe
    c2.dataframe(df1)


    return None


def map_density( data ):
    st.title("Region Overview")
    c1, c2 = st.beta_columns(( 2 ))

    # --------------------------------------------  Map Deensity
    c1.header("Map Density")

    df = data.sample(10)

    # Base Map -  Folium
    density_map = folium.Map(location=[data["lat"].mean(),
                                       data["long"].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker([row["lat"], row["long"]],
                      popup=f"Valor:          {row['price']}\n" \
                            f"Disponível:     {row['date']}\n" \
                            f"Metro Quadrado: {row['sqft_living']}\n" \
                            f"Quartos:        {row['bedrooms']}\n" \
                            f"Banheiros:      {row['bathrooms']}\n" \
                            f"Feito em: {row['yr_built']}").add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Zipcode Price Map - === NEED INSTALL GEOPANDAS ===
    #b3.header("Median Price Density")

    # df = data[["price", "zipcode"]].groupby(  "zipcode" ).mean().reset_index()
    # df = ["ZIPCODE", "PRICE"]

    # df = df.sample( 10 )

    # geofile =  geofile[geofile["ZIPCODE"].isin( df["ZIPCODE"].tolist() )]

    # zipcode_price_map = folium.Map( location=[data["lat"].mean(),
    #                                    data["long"].mean()],
    #                                    default_zoom_start=15 )

    # zipcode_price_map.choropleth( data = df,
    #                              geo_data = geolife,
    #                              columns = ["ZOPCODE","PRICE"],
    #                              key_on = "features.properties.ZIP",
    #                              fill_color = "YlOrRd",
    #                              fill_opacity = 0.7,
    #                              line_opacity = 0.2,
    #                              legend_name = "Mean Price")

    # with c3:
    #    folium_static( zipcode_price_map )

    return None

def commercial_distribuition( data ):
    st.title("Commercial Attributes")

    # Filters - Title
    st.sidebar.title("Commercial Options")

    # --------------------------------------------   Average Price per Year
    st.sidebar.subheader("Select Max Year Built")
    # Filter - Min e Max
    min_year_built = int(data["yr_built"].min())
    max_year_built = int(data["yr_built"].max())
    # Filter - Type
    f_year_built = st.sidebar.slider("Year Built", min_value=min_year_built,
                                                   max_value=max_year_built,
                                                   value=max_year_built)
    # Data Filtering
    st.subheader("Average Price per Year")
    df = data.loc[data["yr_built"] >= f_year_built]
    df = df[["yr_built", "price"]].groupby("yr_built").count().reset_index()
    # Data PLot
    fig = px.line(df, x="yr_built", y="price")
    st.plotly_chart(fig, use_container_width=True)


    # --------------------------------------------   Average Price per Day
    st.subheader("Average Price per Day")
    # Filters - Title
    st.sidebar.subheader("Select Max Date")

    # Filter - Min e Max
    min_date = datetime.strptime(data["date"].min(), "%Y-%m-%d")
    max_date = datetime.strptime(data["date"].max(), "%Y-%m-%d")

    f_date = st.sidebar.slider("Date", min_value=min_date,
                               max_value=max_date,
                               value=max_date)

    # Data Filtering
    data["date"] = pd.to_datetime(data["date"])
    df = data.loc[data["date"] <= f_date]
    df = df[["date", "price"]].groupby("date").mean().reset_index()

    # Plot
    fig = px.line(df, x="date", y="price")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------  Price Distribuition
    st.subheader("Price Distribuition")
    # Filters - Title
    st.sidebar.subheader("Select Max Price")
    # Filter - Attributes
    min_price = int(data["price"].min())
    max_price = int(data["price"].max())
    avg_price = int(data["price"].mean())
    # Filter - Type
    f_price = st.sidebar.slider("Price", min_value= min_price,
                                         max_value= max_price,
                                         value= avg_price)
    # Data Filtering
    df = data.loc[data["price"] <= f_price]
    # Data Plot
    fig = px.histogram( df, x="price", nbins=50 )
    st.plotly_chart( fig, use_container_width=True )

    return None



def attributes_distribuition( data ):
    st.title("House Attributes")

    # Layout Page - 3
    c1, c2 = st.beta_columns(2)

    # --------------------------------------------  House per Bedrooms
    c1.subheader("House per bedrooms")
    # Filter - Title
    st.sidebar.subheader("Select Attributes")
    # Filter - Type
    f_bedrooms = st.sidebar.selectbox("Max number of bedrooms:", sorted(set(data["bedrooms"].unique())))
    # Data Filtering
    df = data[data["bedrooms"] <= f_bedrooms]
    # Data Plot
    fig = px.histogram(df, x="bedrooms", nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------- House per Bathrooms
    c2.subheader("House per bathrooms")
    # Filter - Type
    f_bathrooms = st.sidebar.selectbox("Max number of bathrooms:", sorted(set(data["bathrooms"].unique())))
    # Data Filtering
    df = data[data["bathrooms"] < f_bathrooms]
    # Data Plot
    fig = px.histogram(df, x="bathrooms", nbins=19)
    c2.plotly_chart(fig, use_container_width=True)


    # Layout Page - 3
    c1, c2 = st.beta_columns(2)

    # -------------------------------------------- House per Floors
    c1.subheader("House per floors")
    # Filter - Type
    f_floors = st.sidebar.selectbox("Max number of floors:", sorted(set(data["floors"].unique())))
    # Data Filtering
    df = data[data["floors"] <= f_floors]
    # Data Plot
    fig = px.histogram(df, x="floors", nbins=19)
    c1.plotly_chart(fig, use_container_width=True)


    # -------------------------------------------- House per Water View
    c2.subheader("House per Water View")
    # Filter - Type
    f_waterview = st.sidebar.checkbox("Only Houses with Water View:")
    # Data Filtering
    if f_waterview:
        df = data[data["waterfront"] == 1]
    else:
        df = data.copy()

    # Data Plot
    fig = px.histogram(df, x="waterfront", nbins=10)
    c2.plotly_chart(fig, use_container_width= True)

    return None


#---------- ETL
if __name__ == "__main__":

    # Extration ---------------
    path = "kc_house_data.csv"
    data = get_data( path )
    #url = "https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson"
    #geofile =  get_geofile( url )


    # Transformation -----------------
    data = set_feature( data )

    overview_data( data )

    map_density( data )

    commercial_distribuition( data)

    attributes_distribuition( data )

    # Loading ------------------------












