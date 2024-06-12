import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import warnings
import os
st.set_option('deprecation.showPyplotGlobalUse', False)
# Suppress all warnings
warnings.filterwarnings("ignore")
#COPY PASTING FORMULAS

#constants
G=6.67430e-11 #'''Nm2/kg2'''
m_sun_kg = 1.988e30 #'''kg'''#defined again in functioning
c=299792458 #'''m/s'''
sbc=5.67e-8 #watt/m2K4
pi=3.141592653589
h=6.62607015e-34 #J/Hz
k=1.380649e-23 #m2 kg /(Ks2)
if st.sidebar.checkbox('show constants'):
    st.sidebar.write(' G=6.67430e-11 Nm2/kg2 :s m_sun_kg = 1.988e30 kg :s c=299792458 m/s :s sbc=5.67e-8 watt/m2K4 :s pi=3.141592653589 :s h=6.62607015e-34 J/Hz :s k=1.380649e-23 m2 kg /(Ks2)')
#pre-defining processes to be used later
def integrate_curve(x, y,a=1,b=1):
    integral = 0.0
    for i in range(1,len(x)):
        if y[i-1]!=np.inf and y[i]!=np.inf:
            dx = (x[i] - x[i-1]) #/a
            dy = (y[i] + y[i-1]) #/b
            integral +=  (dy*dx)#*(a*b)
    print(f'print integral is {integral}')
    return integral

def path_of_file(file_name):
    abs_path = os.path.abspath(file_name)
    st.write('file is saved  at: ',abs_path)
def save_data(dataframe):
    file_name = st.text_input("Please enter the name of the CSV file (without extension): ")
    file_name += ".csv"  # Add .csv extension
    if st.button("save data",key="1"):
            dataframe.to_csv(file_name, index=False)
            st.write(f"Data has been successfully saved to {file_name}.")
            st.write(path_of_file(file_name))

def sf(n):
    return len(str(n))

def print_table(aa, bb):
    table = "\n".join([f"{i + 1},  {aa[i]:e},  {bb[i]:e}" for i in range(len(aa))])
    st.write(table)

def print_table_simply(aa, bb):
    table = "\n".join([f"{i + 1},  {aa[i]},  {bb[i]}" for i in range(len(aa))])
    st.write(table)

def name(var):
    for name, value in globals().items():
        if value is var and isinstance(value, list):
            return name
    

def give(zipped, value):
    for key, val in zipped:
        if val == value:
            return key

def plot_multi_curve(x_values, y_values_list, labels):
    xlabel=name(x_values)
    ylabel=name(y_values)
    # Check if number of y_values_list matches the number of labels
    if labels is not None and len(y_values_list) != len(labels):
        raise ValueError("Number of labels must match the number of y value lists.")

    fig, ax = plt.subplots()

    # Plot each curve
    for i, y_values in enumerate(y_values_list):
        label = labels[i] if labels else None
        ax.plot(x_values, y_values, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add title and labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add legend if labels are provided
    if labels:
        ax.legend()
    savethegraph()
    # Return the Matplotlib figure
    return fig

def plot_multi_curve_notlog(x_values, y_values_list,labels):
    

    # Check if number of y_values_list matches the number of labels
    if labels is not None and len(y_values_list) != len(labels):
        raise ValueError("Number of labels must match the number of y value lists.")

    # Plot each curve
    for i, y_values in enumerate(y_values_list):
        label = labels[i] if labels else None
        plt.plot(x_values, y_values, label=label)
    plt.xscale('log')
    plt.yscale('log')
    # Add title and labels
    xlabel=name(x_values)
    ylabel=name(y_values)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Add legend if labels are provided
    if labels:
        plt.legend()
    savethegraph()
    # Show plot
    return fig
def print_constants():
    constants_str = f"boltzman constant k = {k} Hz/K\n" \
                    f"planck's constant h = {h} J/Hz\n" \
                    f"G = {G} Nm2/kg2\n" \
                    f"m_sun_kg = {m_sun_kg} kg\n" \
                    f"c = {c} m/s\n" \
                    f"pi = {pi}\n" \
                    f"sbc or rho = {sbc} (W / (m2 x K4))"
    st.write(constants_str)

def Rs(m):
    r_s = 2*G*m/(c**2)
    return r_s

def temp2(rr):#old formula 
    global t1,t2,t,r
    r=rr*r_s
    t1= 3*G*m_bh_kg*m_dot/(8*pi*sbc)
    t2= (1-(r_i/(r))**0.5)/r**3
    t=(t1*t2)**0.25
    return(t)

def temp(rr):
    global t1,t2,t,r,t_disk
    r=rr*r_s
    t1= (r_i/r)**(3/4)
    t2= (1-(r_i/(r))**0.5)**(1/4)
    t=t_disk*t1*t2
    return(t)


def simpsons_one_third_rule(ff, a, b, n,f):
    #print("in simpson's 1/3 rule")
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even.")

    h = (b - a) / n
    #print('h=',h)
    integral = ff(a) + ff(b)
    #print('integral 1 =',integral)
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * ff(a + i * h)
            #print('integral 2 =',integral)
        else:
            integral += 4 * ff(a + i * h)
            #print('integral 2 =',integral)
    integral *= h/3
    #print('integral final =',integral)
    return integral
def ff(x):
    return (x**(5/3)) / (np.exp(x)-1)
    
def luminosity(f):
    #print('in luminosity function')

    A=64*(pi**2)*(r_i**2)*(k*t_i)**(8/3)
    B=3*(c**2)*(h**(5/3))
    const=(A/B)*f**(1/3)
    integration=simpsons_one_third_rule(ff,(h*f/(k*t_i)),(h*f/(k*t_o)),10000,f)
    lum=const*integration
    return lum
def ff2(r):
    try:
        t1= 3*G*m_bh_kg*m_dot/(8*pi*sbc)    
        return (r) / (np.exp(h*f/(k*((t1*((1-(r_i/(r))**0.5)/r**3))**0.25)))-1)
    except:
        print(f'there is an error at f={f}, r={r}')
def luminosity2(ff):
    #print('in luminosity function')
    global f
    f=ff
    A=16*(pi**2)*h*f**3/c**2
    integration=simpsons_one_third_rule(ff2,r_i+r_s,r_o,10000,f)
    lum=A*integration
    return lum
def plot_log_scale(x_list, y_list,F12=False, show_points=True, interactive=True):
    plt.figure()
    if F12:
        plt.plot([F1,F1],[0,10**22],label='F1')
        plt.plot([F2,F2],[0,10**22],label='F2')
    if show_points:
        plt.scatter(x_list, y_list, marker='.', linestyle='-')
        plt.plot(x_list, y_list, marker='.', linestyle='-')

    else:
        plt.plot(x_list, y_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(str(name(x_list)))
    plt.ylabel(str(name(y_list)))
    plt.legend()
    savethegraph()
    if interactive:
        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.pyplot(plt.gcf())
def plotit(x_list, y_list):
    plt.figure()
    plt.plot(x_list, y_list)
    plt.xlabel(name(x_list))
    plt.ylabel(name(y_list))
    savethegraph()
    st.pyplot()


def intensity(t):
    y=[]
    for v in frequencies:
        i=2*h*(v**(3))/(c**2)*(1/(np.exp(h*v/k*t)-1))
        y.append(i)
    return y

def log10_list_of_lists(input_list):
    # Initialize an empty list to store the results
    output_list = []

    # Iterate through each sublist in the input list
    for sublist in input_list:
        # Initialize an empty sublist to store the log10 of each element
        log_sublist = []
        # Iterate through each element in the sublist
        for num in sublist:
            # Take the log10 of the element and append it to the log_sublist
            log_sublist.append(np.log10(num))
        # Append the log_sublist to the output_list
        output_list.append(log_sublist)

    return output_list

def sum_lists(input_list):
    # Initialize an empty list to store the sums
    sums = []

    # Iterate over the elements of the first sublist to get the length
    length = len(input_list[0])

    # Iterate over each index
    for i in range(length):
        # Initialize the sum for this index
        index_sum = 0

        # Iterate over each sublist in the input list
        for sublist in input_list:
            # Add the corresponding element to the sum
            index_sum += sublist[i]

        # Append the sum to the list of sums
        sums.append(index_sum)

    return sums
def B_lambda(W,T):
    global a,b
    a = 2.0 * h * c**2 /W**5
    b = h * c / (W * k * T)
    return (2 * h * c**2 / W**5) / (np.exp(h * c / (W * k * T)) - 1)

def scaleit(l,a):
    ll=[i*a for i in l]
    return ll

def wavelength_category(wavelength):
    if wavelength < 100:
        return "Vacuum Ultraviolet (VUV)"
    elif 100 <= wavelength < 200:
        return "Deep Ultraviolet (DUV) / Extreme Ultraviolet (EUV)"
    elif 200 <= wavelength < 280:
        return "Far Ultraviolet (FUV)"
    elif 280 <= wavelength < 380:
        return "Ultraviolet (UV)"
    elif 380 <= wavelength < 450:
        return "Violet"
    elif 450 <= wavelength < 495:
        return "Blue"
    elif 495 <= wavelength < 570:
        return "Green"
    elif 570 <= wavelength < 590:
        return "Yellow"
    elif 590 <= wavelength < 625:
        return "Orange"
    elif 625 <= wavelength < 740:
        return "Red"
    else:
        return "Infrared (IR)"

def generate_pattern(n):
    result = []
    current = 1

    while current <= n:
        for i in range(1, 10):
                value = i * current
                if value > n:
                    break
                result.append(value)
        current *= 10
    for i in range(1,r_i_rs):
        try:
            result.remove(i)
        except:
            continue
    return result
def savethegraph():
    name=st.text_input('enter file name')
    if st.button('save'):
        plt.savefig(name)
        st.write(f'Graph saved as {name}')
#---------------------------------------------------------------------------------------------------------
m_bh = st.sidebar.number_input("Mass of the black hole (solar masses)", value=1e8)
m_bh_kg = m_bh * m_sun_kg

# Calculate Schwarzschild radius
r_s = 2 * G * m_bh_kg / c ** 2

# Input for r_i in units of r_s
r_i_rs = st.sidebar.number_input("Value of r_i in units of Schwarzschild radius (r_s)", value=3)
r_i = r_i_rs * r_s

# Input for r_o in units of r_s
r_o_rs = st.sidebar.number_input("Value of r_o in units of Schwarzschild radius (r_s)", value=1e5)
r_o = r_o_rs * r_s

eddington_ratio = st.sidebar.number_input("Eddington ratio", value=0.1)
accretion_efficiency = 0.1  # You can provide a default value or add an input for this if required

# Calculate mass accretion rate
m_dot = (eddington_ratio / accretion_efficiency) * (1.3e31 / c ** 2) * m_bh


#m_dot = eddington_ratio*1.3e31*m_bh_kg/(0.1*(c**2)*m_sun_kg)
t_disk=(3*G*m_bh_kg*m_dot/(8*pi*sbc*(r_i**3)))**0.25
#m_dot=0.6998e27
t_o = temp(r_o_rs)
t_i = temp(r_i_rs+0.1)
F1=k*t_o/h
F2=k*t_i/h
st.sidebar.subheader('parameters')
st.sidebar.text(\
    f"m_bh_kg = {m_bh_kg} kg\n"
    f"r_s = {r_s} m\n"
    f"r_i = {r_i} m\n"
    f"r_o = {r_o} m\n"
    f"m_dot = {m_dot} kg/s\n"
    f"m_dot = {m_dot*31557600/m_sun_kg} solarmass/year\n"
    f"t_o = {t_o} K\n"
    f"t_i = {t_i} K\n"
    f"F1={F1:e} Hz\n"
    f"F2={F2:e} Hz \n"
    f"F2-F1={F2-F1:e} \n")

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


def the_R_vs_T_part(p):
    p+=1
    global radii, temperatures
    st.markdown('# Radius Temprature relationship')
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
    #R_vs_T = dict(zip(radii, temperatures))

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
        st.write(f'The maximum temperature = {tmax:e} K observed at radius {r_tmax:e} Rs.')
    except:
        st.warning('there is some issue in calculating error')
    # Display options for viewing data
    option = st.selectbox("Select4:", ["1) the data table of (R vs T)?", "2) the graph of (R vs T) in logscale",
                                     "3) the graph of (R vs T) without logscale"], key='tvrhere2201{p}')  # Unique key

    if option == "1) the data table of (R vs T)?":
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)
            

    # Plotting the graph for radius vs temperature
    elif option == "2) the graph of (R vs T) in logscale":
        plot_log_scale(radii, temperatures)

    elif option == "3) the graph of (R vs T) without logscale":
        plotit(radii, temperatures)



def the_wavelength_vs_flux_part(p):
    p+=1
    global wavelengths,fluxes_list
    # Defining wavelengths
    wavelengths = np.linspace(1e-9, 3e-7, 50000)  # meters

    # Defining flux at different temperatures
    fluxes_list = []

    temprat = [8e5, 9e5, 10e5]  # kelvin

    for t in temprat:
        fluxes = [np.pi * B_lambda(wavelength, t) for wavelength in wavelengths]
        fluxes_list.append(fluxes)

    option = st.selectbox("Do you want to see the graph of (flux vs wavelength)?", ["Yes", "No"], key='{p}flux_n_wavelength')  # Unique key

    if option == "Yes":
        for i in range(len(temprat)):
            plt.plot(wavelengths * 1e9, scaleit(fluxes_list[i], 1e-4), label=f'{temprat[i]} K')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('*10^4 Flux (W/m^2/nm/sr)')
        plt.title('Planck Function for Different Temperatures')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    st.write("")
    st.write('Peak wavelength should be -:')
    st.write('s.no.  temprature(K)  wavelength(nm)')
    st.table({'Temperature (K)': temprat, 'Wavelength (nm)': [2.897771955e6 / t for t in temprat]})
    
    for i in range(len(temprat)):
        st.write(f'For temperature = {temprat[i]} K')
        f_vs_W = dict(zip(wavelengths, fluxes_list[i]))
        fmax = max(fluxes_list[i])
        wmax = next(key for key, value in f_vs_W.items() if value == fmax) * 1e9
        st.write(f'Maximum flux is {fmax} at wavelength {wmax} nm')
        st.write(f'It falls in the range {wavelength_category(wmax)}')

def the_Frequency_vs_Luminosity_part(p):
    global frequencies, luminosities
    p+=1
    frequencies=sorted([10**n for n in range(1,18)]+[3*10**n for n in range (0,17)])
    # Calculating luminosity at given temperatures
    luminosities = [luminosity(i) for i in frequencies]

    option = st.selectbox("Select2:", ["1) the graph of (F vs L)?", "2) the slopes of (F vs L)?","3) the data table of f vs l"], key="{p}frequency_vs_luminosity")

    if option == "1) the graph of (F vs L)?":
        plot_log_scale(frequencies, luminosities,F12=True)
        L=integrate_curve(frequencies,luminosities,a=1e10,b=1e15) 
        st.info(f"net Luminosity is {L}")
    # To find slopes
    if option == "2) the slopes of (F vs L)?":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        slopes = []
        for i in range(len(frequencies) - 1):
            slope = (log_luminosities[i + 1] - log_luminosities[i]) / (log_frequencies[i + 1] - log_frequencies[i])
            slopes.append(slope)
            st.write(f"Slope {i}: {slope} for point{i+1}=({log_frequencies[i + 1]:e},{log_luminosities[i + 1]:e}) point{i}=({log_frequencies[i]:e},{log_luminosities[i]:e})")
    #to see data
    if option =="3) the data table of f vs l":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        dataset={"frequencies":frequencies,"luminosities":luminosities,"log(frequencies)":log_frequencies,"log(luminosities)":log_luminosities}
        
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)
        
def the_Frequency_vs_Luminosity_part2(p):
    global frequencies, luminosities
    p+=1
    frequencies=sorted([10**n for n in range(1,21)]+[3*10**n for n in range (0,21)]+[7*10**n for n in range (0,21)])

    luminosities=[luminosity2(i) for i in frequencies]

    option = st.selectbox("Select2:", ["1) the graph of (F vs L)?", "2) the slopes of (F vs L)?","3) the data table of f vs l"], key="{p}frequency_vs_luminosity")

    if option == "1) the graph of (F vs L)?":
        plot_log_scale(frequencies, luminosities,F12=True)
        L=integrate_curve(frequencies,luminosities,a=1e10,b=1e15) 
        st.info(f"net Luminosity is {L}")
    # To find slopes
    if option == "2) the slopes of (F vs L)?":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        slopes = []
        for i in range(len(frequencies) - 1):
            slope = (log_luminosities[i + 1] - log_luminosities[i]) / (log_frequencies[i + 1] - log_frequencies[i])
            slopes.append(slope)
            st.write(f"Slope {i}: {slope} for point{i+1}=({log_frequencies[i + 1]:e},{log_luminosities[i + 1]:e}) point{i}=({log_frequencies[i]:e},{log_luminosities[i]:e})")
    #to see data
    if option =="3) the data table of f vs l":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        dataset={"frequencies":frequencies,"luminosities":luminosities,"log(frequencies)":log_frequencies,"log(luminosities)":log_luminosities}
        
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)
        
        
def run(p):
    p+=1
    option_selected = st.sidebar.selectbox("Select:", ["1) the R_vs_T part", "2)(with approx) the Frequency_vs_Luminosity_part",
                                                       "3)(without approx) the Frequency_vs_Luminosity_part","4) the wavelength_vs_flux part"], key="run_selectbox")

    if option_selected == "1) the R_vs_T part":
        p+=1
        the_R_vs_T_part(p)
        
    elif option_selected == "2)(with approx) the Frequency_vs_Luminosity_part":
        p+=1
        the_Frequency_vs_Luminosity_part(p)
    elif option_selected == "3)(without approx) the Frequency_vs_Luminosity_part":
        p+=1
        the_Frequency_vs_Luminosity_part2(p)
    elif option_selected == "4) the wavelength_vs_flux part":
        p+=1
        the_wavelength_vs_flux_part(p)

p=1
run(p)

