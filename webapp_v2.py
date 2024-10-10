#version10
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
global G,m_sun_kg, c, sbc,pi, h,k 
G=6.67430e-11 #'''Nm2/kg2'''
m_sun_kg = 1.988e30 #'''kg'''#defined again in functioning
c=299792458 #'''m/s'''
sbc=5.67e-8 #watt/m2K4
pi=3.141592653589
h=6.62607015e-34 #J/Hz
k=1.380649e-23 #m2 kg /(Ks2)
if st.sidebar.checkbox('show constants'):
    st.sidebar.write(' G=6.67430e-11 Nm2/kg2 :s m_sun_kg = 1.988e30 kg :s c=299792458 m/s :s sbc=5.67e-8 watt/m2K4 :s pi=3.141592653589 :s h=6.62607015e-34 J/Hz :s k=1.380649e-23 m2 kg /(Ks2)')
if st.sidebar.checkbox('show spectrum'):
    st.sidebar.write('(-inf,3e9): Radio :s'\
                     '(3e9,3e12): Microwave :s'\
                     '(3e12,4.3e14): Infrared :s'\
                     '(4.3e14,7.5e14): Visible :s'\
                     '(7.5e14,3e16): Ultraviolet :s'\
                     '(3e16,3e19): X-ray :s'\
                     '(3e19,inf): Gamma-ray :s')

#pre-defining processes to be used later
def integrate_curve(x, y,a=1,b=1):
    integral = 0.0
    for i in range(1,len(x)):
        if y[i-1]!=np.inf and y[i]!=np.inf:
            dx = (x[i] - x[i-1]) #/a
            dy = (np.float128(y[i]) + np.float128(y[i-1])) #/b
            integral +=  (np.float128(dy)*np.float128(dx))#*(a*b)
    print(f'print integral is {integral}')
    return integral
def snip_data(x, y, y1, y2):
    # Convert lists to numpy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)

    mask = (y >= y1) & (y <= y2)

    x_snipped = x[mask]
    y_snipped = y[mask]

    return x_snipped, y_snipped
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

def plot_multi_curve(x_values, y_values_list, labels, xlabel=None, ylabel=None):
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
    if xlabel==None or ylabel==None:
        xlabel=name(x_values)
        ylabel=name(y_values)            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
        
    # Add legend if labels are provided
    if labels:
        ax.legend()
    savethegraph()
    # Return the Matplotlib figure
    return fig

def plot_multi_curve_notlog(x_values, y_values_list,labels,xlabel,ylabel):
    

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
    if xlabel==None or ylabel==None:
        xlabel=name(x_values)
        ylabel=name(y_values)            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    lum=cos_i*A*integration
    return lum
def plot_log_scale(x_list, y_list,temperature=False,spectrumv=False,spectrumf=False,spectrume=False, show_points=True, interactive=True,xlabel='x',ylabel='y'):
    plt.figure()
    fig, ax = plt.subplots()
    p=1
    settings=st.checkbox("Graph settings",value=False)
    
    if spectrumv:
        #to show F1 and F2
        if settings==True:
            F12=st.checkbox("show F1 and F2",)
            if F12:
                plt.plot([F1,F1],[0,10**24],label='F1')
                plt.plot([F2,F2],[0,10**24],label='F2')
        if settings==True or settings== False: 
            #show spectrum lines
            spctrm=True #st.checkbox("show EM spectrum Range",value=True)
            if spctrm:
                plt.fill_between(np.linspace(0,3e9,5),np.linspace(10**24,10**24,5),alpha=0.3,label='radio')
                plt.fill_between(np.linspace(3e9,3e12,5),np.linspace(10**24,10**24,5),alpha=0.3,label='microwave')
                plt.fill_between(np.linspace(3e12,2.99e14,5),np.linspace(10**24,10**24,5),alpha=0.3,label='infrared')
                plt.fill_between(np.linspace(3.01e14,7.5e14,5),np.linspace(10**24,10**24,5),alpha=0.3,label='visible')
                plt.fill_between(np.linspace(7.5e14,3e16,5),np.linspace(10**24,10**24,5),alpha=0.3,label='UV')
                plt.fill_between(np.linspace(3e16,3e19,5),np.linspace(10**24,10**24,5),alpha=0.3,label='X-ray')
                plt.fill_between(np.linspace(3e19,3e30,5),np.linspace(10**24,10**24,5),alpha=0.3,label='Gamma-ray')
    if spectrume or spectrumf:
        if spectrume:
            p=55
        if spectrumf:
            p=45
        if settings:
           
            #show spectrum lines
            spctrm=st.checkbox("show EM spectrum Range",value=True,key=p+1)
            if spctrm:
                
                h1=h/(1.60217663e-19*1e+3)
                plt.fill_between(np.linspace(0,h1*3e9,5),np.linspace(10**51,10**51,5),alpha=0.3,label='radio')
                plt.fill_between(np.linspace(h1*3e9,h1*3e12,5),np.linspace(10**51,10**51,5),alpha=0.3,label='microwave')
                plt.fill_between(np.linspace(h1*3e12,h1*2.9999e14,5),np.linspace(10**51,10**51,5),alpha=0.3,label='infrared')
                plt.fill_between(np.linspace(h1*3.0001e14,h1*7.5e14,5),np.linspace(10**51,10**51,5),alpha=0.3,label='visible')
                plt.fill_between(np.linspace(h1*7.5e14,h1*3e16,5),np.linspace(10**51,10**51,5),alpha=0.3,label='UV')
                plt.fill_between(np.linspace(h1*3e16,h1*3e19,5),np.linspace(10**51,10**51,5),alpha=0.3,label='X-ray')
                plt.fill_between(np.linspace(h1*3e19,h1*3e30,5),np.linspace(10**51,10**51,5),alpha=0.3,label='Gamma-ray')
      
    if show_points:
        plt.scatter(x_list, y_list, marker='.', linestyle='-')
        plt.plot(x_list, y_list, marker='.', linestyle='-')

    else:
        plt.plot(x_list, y_list)
    plt.xscale('log')
    plt.yscale('log')
    if settings:   
        grid=st.checkbox('see grid', key=p+7)
        if grid:
            plt.grid()
    if spectrumv:
        
        x1=1e0
        x2=1e24
        y1=1e0
        y2=1e24
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)

    if spectrume:
        x1=h*1e0/(1.60217663e-19*1e3)
        x2=h*1e24/(1.60217663e-19*1e3)
        y1=1e-4
        y2=1e50
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)
    if spectrumf:
        x1=1e-20
        x2=1e13
        y1=1e-40
        y2=1e2
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)

    if temperature==True:
        x1=0
        x2=r_o_rs*10
        y1=0
        y2=tmax*10
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)

    p+=4
    if settings:
       
        set_range=st.checkbox("set x and y range",key=p+1)
        if set_range:
            x1=st.number_input("lower limit of x", format="%e", value=x1)
            x2=st.number_input("upper limit of x", format="%e", value=x2)
            y1=st.number_input("lower limit of y", format="%e", value=y1)
            y2=st.number_input("upper limit of y", format="%e", value=y2)
            plt.xlim(x1,x2)
            plt.ylim(y1,y2)
    # Add title and labels
    if xlabel=='x' or ylabel=='y':
        xlabel=name(x_values)
        ylabel=name(y_values)            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.legend()
    
    if interactive:
        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.pyplot(plt.gcf())
    savethegraph()
def plotit(x_list, y_list,xlabel='x',ylabel='y'):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(x_list, y_list)
    # Add title and labels
    if xlabel=='x' or ylabel=='y':
        xlabel=name(x_values)
        ylabel=name(y_values)            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
def spectrum_category(frequency):
    
    if frequency < 3e9:  # Radio
        return "Radio"
    elif 3e9 <= frequency < 3e12:  # Microwave
        return "Microwave"
    elif 3e12 <= frequency < 4.3e14:  # Infrared
        return "Infrared"
    elif 4.3e14 <= frequency < 7.5e14:  # Visible
        return "Visible"
    elif 7.5e14 <= frequency < 3e16:  # Ultraviolet
        return "Ultraviolet"
    elif 3e16 <= frequency < 3e19:  # X-ray
        return "X-Ray"
    elif 3e19 <= frequency:  # Gamma-ray
        return "Gamma-ray"

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
    for i in range(1,r_i_rs+1):
        try:
            result.remove(i)
        except:
            continue
    return result
def savethegraph():
    return
    #name=st.text_input('enter file name')
    #if st.button('save'):
     #   plt.savefig(name)
      #  st.write(f'Graph saved as {name}')
def flux_density_nu(nu, T):
    numerator = 2 * h * nu**3 / c**2
    denominator = np.exp(h * nu / (k * T)) - 1
    return numerator / denominator

#---------------------------------------------------------------------------------------------------------
st.sidebar.markdown('# Input values')

m_bh = st.sidebar.number_input("Mass of the black hole (solar masses)", value=1e8,format='%e')
m_bh_kg = m_bh * m_sun_kg

# Calculate Schwarzschild radius
r_s = 2 * G * m_bh_kg / c ** 2

# Input for r_i in units of r_s
r_i_rs = st.sidebar.number_input("Value of r_i in units of Schwarzschild radius (r_s)", value=3,format='%e')
r_i = r_i_rs * r_s

# Input for r_o in units of r_s
r_o_rs = st.sidebar.number_input("Value of r_o in units of Schwarzschild radius (r_s)", value=1e5,format='%e')
r_o = r_o_rs * r_s
# Calculate mass accretion rate

def m_dotf(eddington_ratio,accretion_efficiency):
    return (eddington_ratio/accretion_efficiency) * (1.3e31 / c**2) * m_bh


choice=st.sidebar.selectbox('define either ',['accretion rate','eddington ratio and accretion efficiency'])

if choice =='accretion rate':
    mdot = st.sidebar.number_input("Accretion Rate in solar masses per year", format="%f", value=0.22960933114483387)
    m_dot=mdot*m_sun_kg/(31536000)
    
if choice =='eddington ratio and accretion efficiency':
    st.sidebar.latex(r"\dot{M} = \frac {\epsilon} {\zeta} \frac {1.3 10^{31}}{c^2} M_{BH} \ ,where \ M_{BH} \ is \ in \ solar \ masses")
    eddington_ratio = st.sidebar.number_input(r"Eddington ratio ($\epsilon$)", value=0.1, format="%f")
    accretion_efficiency = st.sidebar.number_input("Accretion efficiency ($\zeta$)", value=0.1, format="%f")
    m_dot=m_dotf(eddington_ratio,accretion_efficiency)

angle_inclination = st.sidebar.number_input("Angle of inclination in degrees", value=0,format='%e')
cos_i=np.cos(angle_inclination)
#m_dot = eddington_ratio*1.3e31*m_bh_kg/(0.1*(c**2)*m_sun_kg)
t_disk=(3*G*m_bh_kg*m_dot/(8*pi*sbc*(r_i**3)))**0.25
t_o = temp(r_o_rs)
t_i = temp(r_i_rs+0.1)
F1=k*t_o/h
F2=k*t_i/h
st.sidebar.subheader('parameters')
st.sidebar.text(\
    f"m_dot = {m_dot} kg/s\n"
    f"m_dot = {m_dot*31557600/m_sun_kg} solarmass/year\n"
    f"m_bh_kg = {m_bh_kg} kg\n"
    f"r_s = {r_s} m\n"
    f"r_i = {r_i} m\n"
    f"r_o = {r_o} m\n"
    f"t_i = {t_i} K\n"
    f"t_o = {t_o} K\n"
    f"cos(i) = {cos_i} \n"
    f"F1={F1:e} Hz\n"
    f"F2={F2:e} Hz \n"
    f"F2/F1={F2/F1:e}")

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


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
        plot_log_scale(radii, temperatures,temperature=True,xlabel="log(radius) (Rs)",ylabel="log(temperature) (K)")

    elif option == "graph of (R vs T) without logscale":
        plotit(radii, temperatures,xlabel="Radius (Rs)",ylabel="Temperature  (K)")

#---------------------------------------------------------------------------------------------------------


def the_wavelength_vs_flux_part(p):
    p+=1
    st.latex(r"B_{\lambda}(\lambda, T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/\lambda k_B T} - 1}")
    global wavelengths,fluxes_list
    # Defining wavelengths
    wavelengths = np.linspace(1e-9, 3e-7, 50000)  # meters

    # Defining flux at different temperatures
    fluxes_list = []

    temprat = [8e5, 9e5, 10e5]  # kelvin

    for t in temprat:
        fluxes = [np.pi * B_lambda(wavelength, t) for wavelength in wavelengths]
        fluxes_list.append(fluxes)

    option = st.selectbox("Options", ["the graph of (flux vs wavelength)", "Dataset"], key='{p}flux_n_wavelength')  # Unique key

    if option == "the graph of (flux vs wavelength)":
        for i in range(len(temprat)):
            plt.plot(wavelengths * 1e9, scaleit(fluxes_list[i], 1e-4), label=f'{temprat[i]} K')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('*10^4 Flux (W/m^2/nm/sr)')
        plt.title('Planck Function for Different Temperatures')
        plt.legend()
        plt.grid(True)
        st.pyplot()
    #DATASET NOT SET 
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
#---------------------------------------------------------------------------------------------------------

def the_frequency_vs_flux_part(p):
    p += 1
    st.latex(r"B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{e^{\frac{h \nu}{k T}} - 1}")
    global frequencies, fluxes_list
    # Defining frequencies
    frequencies = np.linspace(1e14, 1e18, 500)  # in Hz (adjust range as needed)

    # Defining flux at different temperatures (in Kelvin)
    temperatures = [1e5, 2e5, 3e5]

    fluxes_list = []
    for T in temperatures:
        fluxes = [flux_density_nu(nu, T) for nu in frequencies]
        fluxes_list.append(fluxes)

    option = st.selectbox("Do you want to see the graph of (flux vs frequency)?", ["Yes", "No"], key=f'{p}flux_vs_frequency')  # Unique key

    if option == "Yes":
        for i in range(len(temperatures)):
            plt.plot(frequencies / 1e15, fluxes_list[i], label=f'{temperatures[i]} K')
        plt.xlabel('Frequency (10^15 Hz)')
        plt.ylabel('Flux Density (W/m^2/Hz/sr)')
        plt.title('Blackbody Radiation from Accretion Disk at Different Temperatures')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    st.write("")
    st.write('Peak frequency should be -:')
    st.write('s.no.  temperature (K)  frequency (10^15 Hz)')
    st.table({'Temperature (K)': temperatures, 'Frequency (10^15 Hz)': [2.82143937212*(k/(h*1e15))* T for T in temperatures]})

    for i in range(len(temperatures)):
        st.write(f'For temperature = {temperatures[i]} K')
        f_vs_nu = dict(zip(frequencies, fluxes_list[i]))
        fmax = max(fluxes_list[i])
        numax = next(key for key, value in f_vs_nu.items() if value == fmax) / 1e15
        st.write(f'Maximum flux density is {fmax} W/m^2/Hz/sr at frequency {numax} x 10^15 Hz')
        st.write(f'It falls in the range {spectrum_category(fmax)}')
        # Additional information can be added here based on your needs



#---------------------------------------------------------------------------------------------------------

def the_Frequency_vs_Luminosity_part(p):
    global frequencies, luminosities
    p+=1
    frequencies=sorted([10**n for n in range(1,18)]+[3*10**n for n in range (0,17)])
    # Calculating luminosity at given temperatures
    luminosities = [luminosity(i) for i in frequencies]
    st.latex(r'L_\nu = \frac{64 \pi^2 r_i^2 (k T_i)^{8/3}}{3 c^2 h^{5/3}} \nu^{\frac{1}{3}}cosi \int_{x_i}^{x_o}  \frac{x^{5/3}}{e^{x}-1} d x')
    st.latex(r'where,\ \ x(r)=\frac{h \nu}{k T(r)} ')

    L=integrate_curve(frequencies,luminosities,a=1e10,b=1e15)

    

    st.info(f"net Luminosity is {L} ")
    # Storing values of r and t together in R_vs_T
    dataset=pd.DataFrame({"frequency":frequencies,"Luminosity":luminosities})

    # Finding maximum and minimum temperatures
    try:
        lmax = max(luminosities)
        
    except:
        lmin, lmax = 'undetermined', 'undetermined'
    try:     
       # Finding r at maximum temperature
        f_lmax = dataset.loc[dataset['Luminosity'] == lmax, 'frequency'].values[0]
        #display maximum temprature
        st.info(f'The maximum luminosity = {lmax:e} watt/m^2Hz observed at frequency {f_lmax:e} Hz.')
    except:
        st.warning('there is some issue in calculating error')
    st.info(f"{f_lmax:e} Hz lies in {spectrum_category(f_lmax)} ")
   # st.info(f"difference between max frequency and F2 is : {f_lmax - F2 :e} ")
    
    option = st.selectbox("Select:", ["1) the graph of (F vs L)?", "2) the slopes of (F vs L)?","3) the data table of f vs l"], key="{p}frequency_vs_luminosity")

    if option == "1) the graph of (F vs L)?":
        plot_log_scale(frequencies, luminosities,F12=True,xlabel='log(frequencies)',ylabel='log(luminosities)')
        st.latex(r'Unit \ of \ \underline{frequency} \ is \ \bold{ Hz} \ and \ unit \ of \ \underline{luminosity} \ is \ \bold{W m^{-2} Hz^{-1}}')


    # To find slopes
    if option == "2) the slopes of (F vs L)?":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        slopes = []
        for i in range(len(frequencies) - 1):
            slope = (log_luminosities[i + 1] - log_luminosities[i]) / (log_frequencies[i + 1] - log_frequencies[i])
            slopes.append(slope)
        slopes.append('NA')
        d={'slope':slopes,'frequency':frequencies,'Luminosity':luminosities,\
           'log(frequencies)':log_frequencies,\
           'log(Luminosity)':log_luminosities}
        
        dataset=pd.DataFrame(d)
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)    #to see data
    if option =="3) the data table of f vs l":
        log_frequencies = np.log(frequencies)
        log_luminosities = np.log(luminosities)
        d={"frequencies":frequencies,"luminosities":luminosities}
        dataset=pd.DataFrame(d)
        save_data(dataset)
        if st.button("show data"):
            st.table(dataset)
#---------------------------------------------------------------------------------------------------------

def the_Frequency_vs_Luminosity_part2(p):
    global frequencies, luminosities
    p+=1
    #frequencies=sorted([10**n for n in range(1,21)]+[3*10**n for n in range (0,21)]+[7*10**n for n in range (0,21)])
    frequencies=sorted([10**n for n in range(1,23)]+[3*10**n for n in range (0,22)])

    #luminosities=np.array([luminosity2(i) for i in frequencies])
    luminosities=[]
    for i in frequencies:
        Lnew=luminosity2(i)
        if Lnew!=np.inf:
            luminosities.append(Lnew)
        else:
            luminosities.append(0)
        
        
    st.latex(r'L_\nu = \frac{16 \pi^2 h \nu^3}{c^2} cosi \int_{r_i}^{r_o}  \frac{r}{e^{\frac{h \nu}{k T(r)}}-1} d r')
    st.latex(r"where, \ T(r)^4 =\left( \frac {3GM_0\dot{M}} {8 \pi \sigma}\right)\left [\frac{1 - \sqrt{\frac{r_i}{r}}}{r^3} \right]")

    L=integrate_curve(frequencies,luminosities,a=1e10,b=1e15) 
    st.info(f"Net Luminosity is {L}")
    
    #snipped net luminosity
    y1=3e12 #where IR bgins 
    y2=3e16 #where UV ends
    frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities, y1, y2)
    L_snipped = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)
    st.info(f"Net Luminosity from IR(3e12) to UV(3e16)  is {L_snipped}")


    energies=[h*v/(1.60217663e-19*1e3) for v in frequencies]
    EL=[e*v for e,v in zip(luminosities,frequencies)]    
    
    
    # Storing values of r and t together in R_vs_T
    dataset=pd.DataFrame({"frequency":frequencies,"Luminosity":luminosities})    
    option = st.selectbox("Select:", ["1) the graph of (F vs L)?", "2) the slopes of (F vs L)?","3) the data table of f vs l"], key="{p}frequency_vs_luminosity")

    # Finding maximum and minimum temperatures
    try:
        lmax = max(luminosities)
        
    except:
        lmin, lmax = 'undetermined', 'undetermined'
    try:     
       # Finding r at maximum temperature
        f_lmax = dataset.loc[dataset['Luminosity'] == lmax, 'frequency'].values[0]
        #display maximum temprature
    except:
        st.warning('there is some issue in calculating error')


    if option == "1) the graph of (F vs L)?":
        opts=st.checkbox('Other graphs',value=False)
        
        op1=True
        if op1==True:
            st.latex(r'\LARGE{\underline{\bold{L_\nu \ vs \  \nu}}}')
            plot_log_scale(frequencies, luminosities,spectrumv=True,xlabel=r'$log(\nu) \ in \ Hz$',ylabel=r'$log(L_{\nu}) \ in \ W Hz^{-1}$')
            st.info(f'Max $ L_v $ = {lmax:e} watt/m^2Hz observed in {spectrum_category(f_lmax)} region ($ v $ = {f_lmax:.1e} Hz).')

            st.markdown('''<hr style="
                    border: 0;
                    border-top: 3px double white;
                    background: white;
                    margin: 20px 0;" />''', unsafe_allow_html=True)
        if opts:
            
            #op1=st.checkbox(r'$ L_{\nu} \ vs \ \nu $',value=True)
            op2=st.checkbox(r'$ EL_{E} \ vs \ E $',value=True)
            op3=st.checkbox(r'$ EF_{E} \ vs \ E $',value=False)
            if op2==True:
                st.latex(r'\LARGE{\underline{\bold{EL_E \ vs \ E}}}')
                plot_log_scale(energies, EL,spectrume=True, xlabel='log(E) in KeV',ylabel=r'$log(EL_E) \ in \ W $')
                st.markdown('''<hr style="
                        border: 0;
                        border-top: 3px double white;
                        background: white;
                        margin: 20px 0;" />''', unsafe_allow_html=True)
            if op3==True:
                st.latex(r"F_E =\frac {L_E}{4 \pi d^2} \ where \ d \ is \ in \ meters")
                dpsc=st.number_input('enter distance from source in parsecs',value=2.22e3,format='%e')
                d=dpsc*3.0856776e16
                cgs=st.checkbox('cgs unit',value=True)
                EFE=[i/(4*pi*d**2) for i in EL]
                if cgs:
                    EFE_cgs=[i*1e3 for i in EFE]
                    plot_log_scale(energies, EFE_cgs,spectrumf=True, xlabel='log(E) in KeV',ylabel=r'$log(EF_E) \ in \ erg \ s^{-1} \ cm^{-2} $')
                if cgs==False:
                    plot_log_scale(energies, EFE,spectrumf=True, xlabel='log(E) in KeV',ylabel=r'$log(EF_E) \ in \ J \ s^{-1} \ m^{-2} $')
                st.markdown('''<hr style="
                        border: 0;
                        border-top: 3px double white;
                        background: white;
                        margin: 20px 0;" />''', unsafe_allow_html=True)
            
    # To find slopes
    if option == "2) the slopes of (F vs L)?":
        log_frequencies =[ np.log10(float(f)) for f in frequencies]
        log_luminosities =[]
        for lum in luminosities:
            log_lum = np.log10(lum)
            log_luminosities.append(log_lum)

        slopes = []
        for i in range(len(frequencies) - 1):
            slope = (log_luminosities[i + 1] - log_luminosities[i]) / (log_frequencies[i + 1] - log_frequencies[i])
            slopes.append(slope)
        slopes.append(np.nan)
        d={'slope':slopes,'frequency':frequencies,'Luminosity':luminosities,\
           'log(frequencies)':log_frequencies,\
           'log(Luminosity)':log_luminosities}
        
        dataset=pd.DataFrame(d)
        save_data(dataset)
        if st.button("show data"):
            format_str = '{:.2e}'
            d_scientific = {key: [format_str.format(float(value)) for value in values] for key, values in dataset.items()}
            st.table(d_scientific)    #to see data
    if option =="3) the data table of f vs l":
        data={"frequencies":frequencies,"luminosities":luminosities}
        
        dataset=pd.DataFrame(data)
        for column in dataset.columns:
            dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
            
        st.dataframe(dataset, use_container_width=True)
            
#---------------------------------------------------------------------------------------------------------        
def run(p):
    p+=1
    st.markdown('# Spectrum of Standard Accretion Disk')
    option_selected = st.selectbox("Select Property :", ["Luminosity profile",\
                                                         "Temperature Profile"\
                                               ], key="run_selectbox")

    if option_selected == "Temperature Profile":
        p+=1
        the_R_vs_T_part(p)
        
    
    elif option_selected == "Luminosity profile":
        p+=1
        the_Frequency_vs_Luminosity_part2(p)
    
p=1
run(p)
'''Unrequired work now available under extra work'''
def extrawork(p):
    p+=1
    option_selected = st.selectbox("select option:",["Flux profile",\
                                                     "Luminosity profile (with approximation)"],key="show_extras")
    if option_selected == "Flux profile":
        p+=1
        option1=st.checkbox("dependent on frequency")
        if option1:
            the_frequency_vs_flux_part(p)
        option2=st.checkbox("dependent on wavelength")
        if option2:
            the_wavelength_vs_flux_part(p)                                                 
    elif option_selected == "Luminosity profile (with approximation)":
        p+=1
        the_Frequency_vs_Luminosity_part(p)
extra_work = st.checkbox("see extra work done")
if extra_work:
    p=1
    extrawork(p)
st.write(f'save option available when run locally under function savethegraph() line426')
updt=st.checkbox("Update details")
st.markdown("""
<div style='text-align: right;'>
    <p><strong>By Pranjal Sharma</strong></p>
    <p><strong>under guidance of Dr. C. Konar</strong></p>
</div>
""", unsafe_allow_html=True)
if updt:
    st.write(f' \
version 10: added snipped luminosity :s \ 
version 9: added angle of inclination :s \
version 8: added EF_E vs E graph by taking input of d and cgs option in EFE :s \
version 7: added EL_E vs E graph and scaling + grid option :s \
version 6: removed extra work and added spectrum range colours :s\
        ')
