
def the_R_vs_T_part(p):
    p+=1
    global radii, temperatures, tmax
    st.markdown('# Radius-Temperature relationship')
    st.latex(r"where, T(r)^4 =\left( \frac {3GM_{BH}\dot{M}} {8 \pi \sigma}\right)\left [\frac{1 - \sqrt{\frac{r_i}{r}}}{r^3} \right]")
    # Creating list of radii
    radii = generate_pattern(r_o_rs)

    # Defining temperature at different radii
    temperatures = []
    for i in radii:
        try:
            t = temp(i)
            temperatures.append(t)
        except:
            temperatures.append(0)

    # Storing values of r and t together in R_vs_T
    dataset=pd.DataFrame({"radius in rs":radii,"temperatures":temperatures})

    # Finding maximum and minimum temperatures
    try:
        tmax = max(temperatures)
        tmin = min(temperatures)
    except:
        tmin, tmax = 'undetermined', 'undetermined'
    try:     
       # Finding r at maximum temperature
        r_tmax = dataset.loc[dataset['temperatures'] == tmax, 'radius in rs'].values[0]
        #display maximum temprature
        st.info(f'The maximum temperature = {tmax:e} K observed at radius {r_tmax:e} Rs.')
        st.info(f'The minimum temperature = {temp(r_o_rs)} K ')
    except:
        st.warning('there is some issue in calculating error')

    # Display options for viewing data
    option = st.selectbox("Select:", ["graph of (R vs T) in logscale",\
                                      "data table of (R vs T)?",\
                                     "graph of (R vs T) without logscale"], key='tvrhere2201{p}')  # Unique key

    if option == "data table of (R vs T)?":
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)
            

    # Plotting the graph for radius vs temperature
    elif option == "graph of (R vs T) in logscale":
        plot_log_scale(radii, temperatures,0,r_o_rs,tmin,tmax,temperature=True,xlabel="log(radius) (Rs)",ylabel="log(temperature) (K)")

    elif option == "graph of (R vs T) without logscale":
        plotit(radii, temperatures,xlabel="Radius (Rs)",ylabel="Temperature  (K)")
