"version15"
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import warnings
import os

#----------------------------------SECTION 1----------------------------------------------------------
# Suppress all warnings
st.set_option('deprecation.showPyplotGlobalUse', False) 
warnings.filterwarnings("ignore")

#set version
st.sidebar.info(" ## Version 15")

#----------------------------------SECTION 2----------------------------------------------------------

#constants
global G,m_sun_kg, c, sbc,pi, h,k 
G=6.67430e-11 #'''Nm2/kg2'''
m_sun_kg = 1.988e30 #'''kg'''#defined again in functioning
c=299792458 #'''m/s'''
sbc=5.67e-8 #watt/m2K4
pi=3.141592653589
h=6.62607015e-34 #J/Hz
k=1.380649e-23 #m2 kg /(Ks2)

FO1=3.2928e15
FO2=8.4922e15
if st.sidebar.checkbox('show constants'):
    st.sidebar.write('EO1=13.618 eV :s FO1=3.2928e15 Hz :s EO2=35.121 eV :s FO2=8.4922e15 Hz :s G=6.67430e-11 Nm2/kg2 :s m_sun_kg = 1.988e30 kg :s c=299792458 m/s :s sbc=5.67e-8 watt/m2K4 :s pi=3.141592653589 :s h=6.62607015e-34 J/Hz :s k=1.380649e-23 m2 kg /(Ks2)')

#----------------------------------SECTION 3----------------------------------------------------------

#pre-defining processes to be used later

#trapezoid rule integration
def integrate_curve(x, y,a=1,b=1): 
    integral = 0.0
    for i in range(1,len(x)):
        if y[i-1]!=np.inf and y[i]!=np.inf:
            dx = (x[i] - x[i-1]) #/a
            dy = (np.float128(y[i]) + np.float128(y[i-1])) #/b
            integral +=  (np.float128(dy)*np.float128(dx))#*(a*b)
    return integral

#to take data upto a certain value of x and y 
def snip_data(x, y, x1, x2):
    x = np.array(x)
    y = np.array(y)
    mask = (x >= x1) & (x <= x2)
    x_snipped = x[mask]
    y_snipped = y[mask]

    return x_snipped, y_snipped

#doesnt work through webapps but works on localhost
def save_data(dataframe):
    file_name = st.text_input("Please enter the name of the CSV file (without extension): ")
    file_name += ".csv"  # Add .csv extension
    if st.button("save data",key="1"):
            dataframe.to_csv(file_name, index=False)
            st.write(f"Data has been successfully saved to {file_name}.")
            #st.write(path_of_file(file_name))

#find schwarzchild radius
def Rs(m):
    r_s = 2*G*m/(c**2)
    return r_s

#temperature
def temp(rr):
    global t1,t2,t,r,t_disk
    r=rr*r_s
    t1= (r_i/r)**(3/4)
    t2= (1-(r_i/(r))**0.5)**(1/4)
    t=t_disk*t1*t2
    return(t)

#integrating method
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

#defining funtion to be integrated
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


#THE ULTIMATE PLOTTER

def plot_log_scale(x_list, y_list,xo,xn,yo,yn,temperature=False,spectrumv=False,spectrume=False, show_points=True, interactive=True,xlabel='x',ylabel='y'):
    plt.figure()
    fig, ax = plt.subplots()
    p=np.random.randint(100)
    settings=st.checkbox("Graph settings",value=True,key=p*100)
    
    if spectrumv:
        #fill bg
        plt.fill_between(np.linspace(0,3e9,5),np.linspace(10**24,10**24,5),alpha=0.3,label='radio')
        plt.fill_between(np.linspace(3e9,3e12,5),np.linspace(10**24,10**24,5),alpha=0.3,label='microwave')
        plt.fill_between(np.linspace(3e12,2.99e14,5),np.linspace(10**24,10**24,5),alpha=0.3,label='infrared')
        plt.fill_between(np.linspace(3.01e14,7.5e14,5),np.linspace(10**24,10**24,5),alpha=0.3,label='visible')
        plt.fill_between(np.linspace(7.5e14,3e16,5),np.linspace(10**24,10**24,5),alpha=0.3,label='UV')
        plt.fill_between(np.linspace(3e16,3e19,5),np.linspace(10**24,10**24,5),alpha=0.3,label='X-ray')
        plt.fill_between(np.linspace(3e19,3e30,5),np.linspace(10**24,10**24,5),alpha=0.3,label='Gamma-ray')

        #to show F1 and F2
        if settings==True:
            F12=st.checkbox("show F1 and F2 (The point of change in slope of curve)",)
            if F12:
                plt.plot([F1,F1],[0,10**24],label='F1')
                plt.plot([F2,F2],[0,10**24],label='F2')
        
                plt.plot()
    
            FO12=st.checkbox("show frequency for 1st and second ionisation of oxygen",value=True,key=p*100+1)
            if FO12:
                plt.plot([FO1,FO1],[0,10**24],label='First ionisation of oxygen')
                plt.plot([FO2,FO2],[0,10**24],label='second ionisation of oxygen')
    if spectrume :
        p=220202
        
        #show spectrum lines
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
    plt.xlim(xo,xn)
    plt.ylim(yo,yn)      





    if temperature==True:
        x1=0
        x2=r_o_rs*10
        y1=0
        y2=tmax*10
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)

    p+=4
    if settings:
        set_range=st.checkbox("set x and y range",key=p+1,value=False)
        if set_range:
            x1=st.number_input("lower limit of x", format="%e", value=xo)
            x2=st.number_input("upper limit of x", format="%e", value=xn)
            y1=st.number_input("lower limit of y", format="%e", value=yo)
            y2=st.number_input("upper limit of y", format="%e", value=yn)
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

#SIMPLER PLOTTER
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

#TO FIND INTENSITY
def intensity(t):
    y=[]
    for v in frequencies:
        i=2*h*(v**(3))/(c**2)*(1/(np.exp(h*v/k*t)-1))
        y.append(i)
    return y

#FREQUENCY SPECTRUM RANGE
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

#PATTERN OF 1,2,3...10,20,30...100,200,300... SO ON
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
    for i in range(1,int(r_i_rs+1)):
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

#-----------------------------------SECTION 4----------------------------------------------------------------------
#TAKE INPUTS
st.sidebar.markdown('# Input values')

m_bh = st.sidebar.number_input("Mass of the black hole (solar masses)", value=1e8,format='%e')
m_bh_kg = m_bh * m_sun_kg

# Calculate Schwarzschild radius
r_s = 2 * G * m_bh_kg / c ** 2

# Input for r_i in units of r_s
r_i_rs = st.sidebar.number_input("Value of r_i in units of Schwarzschild radius (r_s)", value=3e0,format='%e')
r_i = r_i_rs * r_s

# Input for r_o in units of r_s
r_o_rs = st.sidebar.number_input("Value of r_o in units of Schwarzschild radius (r_s)", value=1e5,format='%e')
r_o = r_o_rs * r_s

# Calculate mass accretion rate
def m_dotf(eddington_ratio,accretion_efficiency):
    return (eddington_ratio/accretion_efficiency) * (1.3e31 / c**2) * m_bh


choice=st.sidebar.selectbox('Define either ',['Accretion rate','Eddington ratio and accretion efficiency'])

if choice =='Accretion rate':
    mdot = st.sidebar.number_input("Accretion Rate in solar masses per year", format="%f", value=0.22960933114483387)
    m_dot=mdot*m_sun_kg/(31536000)
    
if choice =='Eddington ratio and accretion efficiency':
    st.sidebar.latex(r"\dot{M} = \frac {\epsilon} {\zeta} \frac {1.3 10^{31}}{c^2} M_{BH} \ ,where \ M_{BH} \ is \ in \ solar \ masses")
    eddington_ratio = st.sidebar.number_input(r"Eddington ratio ($\epsilon$)", value=1e-1, format="%e")
    accretion_efficiency = st.sidebar.number_input("Accretion efficiency ($\zeta$)", value=1e-1, format="%e")
    m_dot=m_dotf(eddington_ratio,accretion_efficiency)

angle_inclination = st.sidebar.number_input("Angle of inclination in degrees", value=0,format='%e')
cos_i=np.cos(angle_inclination)
#m_dot = eddington_ratio*1.3e31*m_bh_kg/(0.1*(c**2)*m_sun_kg)
t_disk=(3*G*m_bh_kg*m_dot/(8*pi*sbc*(r_i**3)))**0.25
t_o = temp(r_o_rs)
t_i = temp(r_i_rs+0.1)
F1=k*t_o/h
F2=k*t_i/h
st.sidebar.subheader('Parameters')
st.sidebar.write(\
    r"$ \dot{M} $"+f" = {m_dot:.4e} " + r"$ kg s^{-1} " +
    r"$ :s "+
    r"$ \dot{M} $"+f" = {m_dot*31557600/m_sun_kg:.4e} "+ r"$  M_{\odot} yr^{-1} $"
    r" :s"+
    r"$ M_{\bullet}" f"= {m_bh_kg} kg" +
    r"$ :s"+
    r"$ R_s "+f" = {r_s:.4e} m"
    r"$ :s"+
    r"$ R_i"+f" = {r_i:.4e} m"+
    r"$ :s"+
    r"$ R_o"+f" = {r_o:.4e} m"
    r"$ :s"+
    r"$ T_i"+f" = {t_i:.4e} K"
    r"$ :s"+
    r"$ T_o $"+f" = {t_o:.4e} K ")
st.sidebar.write(f" cos(i) = {cos_i}"
    r" :s ",
    f" F1={F1:e} Hz",
    r" :s "
    f"F2={F2:e} Hz"
    r" :s "
    f"F2/F1={F2/F1:e}")

#----------------------------------SECTION 5-----------------------------------------------------------------------

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

def the_Frequency_vs_Luminosity_part2(p):
    global frequencies, luminosities
    
    #creating a list of frequencies based on large or small data option
    p+=1
    col1,col2,col3 = st.columns([0.2,0.1,1])
    with col1:
        st.write("Large dataset")
    with col2:
        a=st.toggle(" ",value=True)
    with col3:
        st.write("Mindful subset of dataset")
    if a:
        frequencies=sorted([10**n for n in range(1,23)]+ \
                       [3*10**n for n in range (0,22)]+ \
                           list(np.linspace(1e-5*F1,100*F2,40))+ \
                           list(np.linspace(3e12, 4.3e14,20))+ \
                           list(np.linspace(4.3e14, 7.5e14,20))+ \
                           list(np.linspace(7.5e14, 3e16,20)))
    else:
        frequencies=sorted([10**n for n in range(1,23)]+ \
                           [2*10**n for n in range (0,22)]+ \
                           [4*10**n for n in range (0,22)]+ \
                           [6*10**n for n in range (0,22)]+ \
                           [7*10**n for n in range (0,22)]+ \
                           [9*10**n for n in range (0,22)]+ \
                           [8*10**n for n in range (0,22)]+ \
                           list(np.linspace(1e-5*F1,100*F2,40))+ \
                           list(np.linspace(3e12, 4.3e14,20))+ \
                           list(np.linspace(4.3e14, 7.5e14,20))+ \
                           list(np.linspace(7.5e14, 3e16,20)))

    #finding corresponding luminosity density for selected frequency list                      
    luminosities=[]
    for i in frequencies:
        Lnew=luminosity2(i)
        if Lnew!=np.inf:
            luminosities.append(Lnew)
        else:
            luminosities.append(0)

    #writing formulas using latex
            
    col1,col2=st.columns([2,1.5])
    with  col1:
        st.latex(r" \dot{M} = \frac {\epsilon} {\zeta} \frac {1.3 \times 10^{31}}{c^2} \frac{M_{\bullet}}{M_{\odot}} ")    
        st.latex(r"T(r)^4 =\left( \frac {3GM_{\bullet}\dot{M}} {8 \pi \sigma}\right)\left [\frac{1 - \sqrt{\frac{r_i}{r}}}{r^3} \right] \ \ K^4")
        st.latex(r'L_\nu = \frac{16 \pi^2 h \nu^3}{c^2} cosi \int_{r_i}^{r_o}  \frac{r}{e^{\frac{h \nu}{k T(r)}}-1} d r \ \ W Hz^{-1}')
    
    with col2:
        st.write(r"$ Where, $")
        st.write(r"$ \epsilon \rightarrow Eddington \ ratio $")
        st.write(r"$ \zeta \rightarrow Accretion \ efficiency $")
        st.write(r"$ M_{\bullet} \rightarrow Mass \ of \ black \ hole $")
        st.write(r"$ \dot{M} \rightarrow Accretion \ rate $")
        st.write(r"$ \sigma \rightarrow  Stefan \ Bolzsmann \ constant $")
        st.write(r"$ h \rightarrow  Planck's \ constant $")

    if st.checkbox(r"Derivation of $ \dot{M} $",value=True):
        st.latex(r"L = \zeta \dot{M} c^2")
        st.latex(r"L = \epsilon L_{Edd}")
        st.latex(r"\dot{M} = \frac{L}{\zeta c^2}")
        st.latex(r"\dot{M} = \frac{\epsilon L_{Edd}}{\zeta c^2}")
        st.latex(r" \dot{M} = \frac {\epsilon} {\zeta} \frac {1.3 \times 10^{31}}{c^2} \frac{M_{\bullet}}{M_{\odot}} ")    
        
    #spectrum range :
    if st.checkbox('show spectrum'):
        data = {
        "Category":
            ["Radio",
            "Microwave",
            "Infrared",
            "Visible",
            "Ultraviolet",
            "X-ray",
            "Gamma-ray"],
        "Frequency Range (Hz)":
        [ "(-inf, 3e9)",
            "(3e9, 3e12)",
            "(3e12, 4.3e14)",
            "(4.3e14, 7.5e14)",
            "(7.5e14, 3e16)",
            "(3e16, 3e19)",
            "(3e19, inf)"],
        "Wavelength Range (m)":
        [ f"(above {c/3e9:.3e})",
            f"({c/3e9:.3e}, {c/3e12:.3e})",
            f"({c/3e12:.3e}, {c/4.3e14:.3e})",
            f"({c/4.3e14:.3e}, {c/7.5e14:.3e})",
            f"({c/7.5e14:.3e}, {c/3e16:.3e})",
            f"({c/3e16:.3e}, {c/3e19:.3e})",
            f"({c/3e19:.3e}, 0)"]}
        st.table(data)

    #integrating the luminosity density curve wuth respect to frequency to get Luminosity
    L=integrate_curve(frequencies,luminosities,a=1e10,b=1e15)
    st.info(f"Bolometric Luminosity L = {L} Watts")

    #integrating luminosity density for a fixed range of frequency
    if st.checkbox("For custom range of frequency",value=True):
        #snipped net luminosity
        col1,col2,col3,col4,col5 =st.columns([0.3,0.3,0.1,0.3,0.4])
        with col1:
            ""
            ""
            "Net luminosity from"
        with col2:
            x1=st.number_input("",value=FO1,format='%e')  
        with col3:
            ""
            ""
            "Hz to"
        with col4:
            x2=st.number_input("",value=FO2,format='%e') 
        with col5:
            ""
            ""
            "Hz"
        frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities, x1, x2)
        L_snipped = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)
        st.success(f"Net Luminosity from "+ r'$ \nu_1  =  $'+f" {x1:.2e} Hz ({spectrum_category(x1)}) to " + r"$ \nu_2  =  $"+ f"{x2:.2e} Hz ({spectrum_category(x2)})  is {L_snipped} Watts")
        st.success(f"Average Luminosity density "+\
               r'$ \bar{L} =\frac{ {\int_{\nu_1}^{\nu_2}}{L_\nu} d\nu} {\nu_2-\nu_1} = $' +f" {L_snipped/(x2-x1):e} Watts/Hz")

    else:
        frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities, 3e12, 4.3e14)
        L_IR = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)

        frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities, 4.3e14, 7.5e14)
        L_visible = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)

        frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities,7.5e14, 3e16)
        L_UV = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)

        frequencies_snipped,luminosities_snipped = snip_data(frequencies, luminosities,3e16, 3e19)
        L_xray = integrate_curve(frequencies_snipped,luminosities_snipped,a=1e10,b=1e15)

        df=pd.DataFrame({"Range":["IR","Visible","UV","X-ray"],
                        "Frequency":['3e12 Hz to 4.3e14 Hz','4.3e14 Hz to 7.5e14 Hz','7.5e14 Hz to 3e16 Hz'," 3e16 Hz to 3e19 Hz"],
                         "Luminosity(watts)":[L_IR,L_visible,L_UV,L_xray]})
        # Convert float128 to float64 in DataFrame
        df['Luminosity(watts)'] = df['Luminosity(watts)'].astype(np.float64)
        df = df.applymap(lambda x: f'{x:.2e}' if isinstance(x, (int, float)) else x)

        st.table(df)

    #changing frequency to KeV and EL= frequency * luminosity density 
    energies=[h*v/(1.60217663e-19*1e3) for v in frequencies]
    EL=[e*v for e,v in zip(luminosities,frequencies)]    
    
    
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
    except:
        st.warning('there is some issue in calculating error')

    st.header("SELECT OUTPUT FORMAT")
    opts=st.selectbox('',['cloudy default units','frequency vs luminosity density',\
                                      'Energy vs Energy flux','Energy vs flux',\
                                      'frequency vs Energy flux'])
    
    if opts=='cloudy default units':
        xo=1e1
        xn=1e20
        yo=1e-30
        yn=1e10
        st.latex(r"\nu F_\nu =\nu \frac {L_\nu}{4 \pi d^2} \ where \ d \ is \ in \ meters")
        dpsc=st.number_input('enter distance from source in parsecs',value=1,format='%e')
        d=dpsc*3.0856776e16            
        nuFnu=[nu*L/(4*pi*d**2) for nu,L in zip(frequencies,luminosities)]
                 
        freq_Ryd=[i/3.28984196e15 for  i in frequencies]
        nuFnu_cgs =[i*1e3 for i in nuFnu]
        
        plot_log_scale(frequencies, nuFnu_cgs,xo,xn,yo,yn,spectrumv=True, xlabel=r'$log(\nu) in Hz$',ylabel=r'$log(\nu F_{\nu}) (erg/(s cm^2) $')
        st.header("How to make datafile cloudy friendly")
        col1,col2=st.columns([2,1])
        with col1:
            st.write("1. Download data file.")
            st.write("2. Open in excel to remove first column of indexes")
            st.write("3. After the first entry write nuFnu.")
            st.write("4. Add 3 or more stars to mark end of file ***")
        with col2:
            st.write("example:")
            st.write("#datafile"," \n"," 1.4 1e11 nuFnu"," \n","1.5 1e10")
            st.text("... \n*************")
 
        data={"log(freq) (Ryd)":np.log10(freq_Ryd),"nuFnu (erg/(s cm^2))":nuFnu_cgs}    
        dataset=pd.DataFrame(data)
        dataset["nuFnu (erg/(s cm^2))"] = dataset["nuFnu (erg/(s cm^2))"].apply(lambda x: '{:.2e}'.format(x))
        st.dataframe(dataset, use_container_width=True)


    
    if opts=='frequency vs luminosity density':
        xo=1e0 # default lower limit of x
        xn=1e24# default upper limit of x
        yo=1e0# default lower limit of y
        yn=1e24# default upper limit of y
        
        st.latex(r'\LARGE{\underline{\bold{L_\nu \ vs \  \nu}}}')

        plot_log_scale(frequencies, luminosities,xo,xn,yo,yn,spectrumv=True,xlabel=r'$log(\nu) \ in \ Hz$',ylabel=r'$log(L_{\nu}) \ in \ W Hz^{-1}$')

        st.info(r'Max $ L_\nu $ = ' + f'{lmax:e} '+ r' $W m^{-2} Hz^{-1}$' + f'observed in {spectrum_category(f_lmax)} region at ' +r'$ \nu $'+ f" = {f_lmax:.1e} Hz")

        data={"frequencies (Hz)":frequencies,"luminosities (W/Hz)":luminosities}
        dataset=pd.DataFrame(data)
        for column in dataset.columns:
            dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
        st.dataframe(dataset, use_container_width=True)            

        st.markdown('''<hr style="
                border: 0;
                border-top: 3px double white;
                background: white;
                margin: 20px 0;" />''', unsafe_allow_html=True)

    if opts=='Energy vs Energy flux':
        st.latex(r'\LARGE{\underline{\bold{EL_E \ vs \ E}}}')
        xo=h*1e0/(1.60217663e-19*1e3)
        xn=h*1e24/(1.60217663e-19*1e3)
        yo=1e-4
        yn=1e50
        plot_log_scale(energies, EL,xo,xn,yo,yn,spectrume=True, xlabel='log(E) in KeV',ylabel=r'$log(EL_E) \ in \ W $')

        data={"Energy (keV)":energies,"energy flux (W)":EL}
        dataset=pd.DataFrame(data)
        for column in dataset.columns:
            dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
        st.dataframe(dataset, use_container_width=True)

        st.markdown('''<hr style="
            border: 0;
            border-top: 3px double white;
            background: white;
            margin: 20px 0;" />''', unsafe_allow_html=True)

    if opts=='Energy vs flux':

        
        st.latex(r"F_E =\frac {L_E}{4 \pi d^2} \ where \ d \ is \ in \ meters")
        
        dpsc=st.number_input('enter distance from source in parsecs',value=2.22e3,format='%e')
        d=dpsc*3.0856776e16 #changing distance to m
        
        cgs=st.checkbox('cgs unit',value=True)
        EFE=[i/(4*pi*d**2) for i in EL]
        if cgs:
            xo=1e-17
            xn=1e16
            yo=1e-40
            yn=1e2
            EFE_cgs=[i*1e3 for i in EFE]
            energies_eV=[i*1e3 for i in energies]
            plot_log_scale(energies_eV, EFE_cgs,xo,xn,yo,yn,spectrume=True, xlabel='log(E) in eV',ylabel=r'$log(EF_E) \ in \ erg \ s^{-1} \ cm^{-2} $')
            data={"Energy (eV)":energies_eV,"Flux (erg/(s cm^2))":EFE_cgs}
            dataset=pd.DataFrame(data)
            for column in dataset.columns:
                dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
            st.dataframe(dataset, use_container_width=True)

        if cgs==False:
            xo=1e-20
            xn=1e13
            yo=1e-40
            yn=1e2
            plot_log_scale(energies, EFE,xo,xn,yo,yn,spectrume=True, xlabel='log(E) in KeV',ylabel=r'$log(EF_E) \ in \ J \ s^{-1} \ m^{-2} $')
            data={"Energy (keV)":energies,"Flux (J/(s m^2))":EFE}
            dataset=pd.DataFrame(data)
            for column in dataset.columns:
                dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
            st.dataframe(dataset, use_container_width=True)

        st.markdown('''<hr style="
                border: 0;
                border-top: 3px double white;
                background: white;
                margin: 20px 0;" />''', unsafe_allow_html=True)

    
    if opts=='frequency vs Energy flux':
        xo=1e6
        xn=1e20
        yo=1e-30
        yn=1e-9
        st.latex(r"\nu L_\nu =\nu \frac {L_\nu}{4 \pi d^2} \ where \ d \ is \ in \ meters")
        dpsc=st.number_input('enter distance from source in parsecs',value=1620.3e6,format='%e')
        d=dpsc*3.0856776e16            
        nuLnu=[nu*L/(4*pi*d**2) for nu,L in zip(frequencies,luminosities)]
        plot_log_scale(frequencies, nuLnu,xo,xn,yo,yn,spectrumv=True, xlabel=r'$log(\nu) in Hz$',ylabel=r'$log(\nu L_{\nu}) (watts) $')

        if st.checkbox("View data(Hz vs watts)",key="data g4"):
            data={r"frequencies (Hz)":frequencies,r"$\nu L_{\nu} (W)":nuLnu}
            
            dataset=pd.DataFrame(data)
            for column in dataset.columns:
                dataset[column] = dataset[column].apply(lambda x: '{:.2e}'.format(x))
            
            st.dataframe(dataset, use_container_width=True)

        if st.checkbox("View data (Ryd vs erg/s)",key="data g5",value=True):
            
            freq_Ryd=[i/3.28984196e15 for  i in frequencies]
            nuLnu_cgs =[i*1e3 for i in nuLnu]
            data={r"frequencies (Hz)":freq_Ryd,r"$\nu L_{\nu} erg/s":nuLnu_cgs}
            
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

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# conversion calculator 
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
converter = st.checkbox("Conversion calculator", value= False)
if converter :
    conversion_to_si = {
        # Frequency units to Hertz (Hz)
        "Hertz (Hz)": 1,
        "Kilohertz (kHz)": 1e3,
        "Megahertz (MHz)": 1e6,
        "Gigahertz (GHz)": 1e9,
        
        # Wavelength units to Meters (m)
        "Meters (m)": 1,
        "Centimeters (cm)": 0.01,
        "Nanometers (nm)": 1e-9,
        "Picometers (pm)": 1e-12,
        "Ångström (Å)": 1e-10,
        
        # Energy units to Joules (J)
        "Joules (J)" : 1,
        "Ergs (Erg)": 1e-7,
        "Electron Volts (eV)": 1.60218e-19,
        "Kiloelectron Volts (keV)": 1.60218e-16, 
    }


    # Categories
    frequency_units = list(conversion_to_si.keys())[:4]
    wavelength_units = list(conversion_to_si.keys())[4:9]
    energy_units = list(conversion_to_si.keys())[9:]

    # Input for value and units
    col1, col2, col3, col4 = st.columns(4)

    #taking input value and unit
    with col1:
        input_value = st.number_input("Input Value:", format="%e")
    with col2:
        input_unit = st.selectbox("Input Unit:", list(conversion_to_si.keys()))

    #changing input value si version of input unit
    if input_unit in frequency_units:
        si_input="Hz"
        output = input_value*conversion_to_si.get(input_unit)
    elif input_unit in wavelength_units:
        si_input="m"
        output = input_value*conversion_to_si.get(input_unit)
    elif input_unit in energy_units:
        si_input="J"
        output = input_value*conversion_to_si.get(input_unit)

    #selecting output unit
    with col4:
        output_unit = st.selectbox("Output Unit:", list(conversion_to_si.keys()))

    #changing to output unit 
    if output_unit in frequency_units:
        if input_unit in frequency_units:
            output = output / conversion_to_si.get(output_unit)

        elif input_unit in wavelength_units:
            output = (c/output) / conversion_to_si.get(output_unit)

        elif input_unit in energy_units:
            output = (output/h) / conversion_to_si.get(output_unit)

    if output_unit in wavelength_units:
        if input_unit in frequency_units:
            output = (c/output) / conversion_to_si.get(output_unit)

        elif input_unit in wavelength_units:
            output = (output) / conversion_to_si.get(output_unit)

        elif input_unit in energy_units:
            output = (h*c/output) / conversion_to_si.get(output_unit)

    if output_unit in energy_units:
        if input_unit in frequency_units:
            output = (h*output) / conversion_to_si.get(output_unit)

        elif input_unit in wavelength_units:
            output = (h*c/output) / conversion_to_si.get(output_unit)

        elif input_unit in energy_units:
            output = (output) / conversion_to_si.get(output_unit)

    #display output
    with col3:
        st.success(f'output value :s {output:.2e}')
st.markdown('''<hr style="
                    border: 0;
                    border-top: 3px double white;
                    background: white;
                    margin: 20px 0;" />''', unsafe_allow_html=True)
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#By line
st.markdown("""
<div style='text-align: centre ;'>
    <p><strong>By Pranjal Sharma</strong></p>
    <p><strong>Under the guidance of Dr. C. Konar</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown('''<hr style="
                    border: 0;
                    border-top: 3px double white;
                    background: white;
                    margin: 20px 0;" />''', unsafe_allow_html=True)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

updt=st.checkbox("Update details")
if updt:
    st.write("""
    version 15: added cloudy friendly data setup option 
    version 14: changed to multiple graph select box, removed the option to check slope, now need to define
                x and y range in plotlogscale itself, bg colour selected based on spectrumv or spectrume
    version 13: Using symbols to show parameters + added m_dot equation + removed extra work  :s
    version 12: improved string format and changed input frequency pattern, corrected snipped integration :s
    version 11: added conversion calculator + custom freq selection while calculating net luminosity :s 
    version 10: added snipped luminosity :s
    version 9: added angle of inclination :s
    version 8: added EF_E vs E graph by taking input of d and cgs option in EFE :s
    version 7: added EL_E vs E graph and scaling + grid option :s
    version 6: added spectrum range colours
    """)
