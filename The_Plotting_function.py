#the_ultimate_plotter
def plot_log_scale(x_list, y_list,xo,xn,yo,yn,temperature=False,spectrumv=False,spectrume=False, show_points=True, interactive=True,xlabel='x',ylabel='y'):
    plt.figure()
    fig, ax = plt.subplots()
    p=np.random.randint(100)
    settings=st.checkbox("Graph settings",value=False,key="Graph_Settings")
      
    if spectrumv:
        #fill bg
        plt.fill_between(np.linspace(0,3e9,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='radio')
        plt.fill_between(np.linspace(3e9,3e12,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='microwave')
        plt.fill_between(np.linspace(3e12,2.99e14,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='infrared')
        plt.fill_between(np.linspace(3.01e14,7.5e14,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='visible')
        plt.fill_between(np.linspace(7.5e14,3e16,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='UV')
        plt.fill_between(np.linspace(3e16,3e19,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='X-ray')
        plt.fill_between(np.linspace(3e19,3e30,5),np.linspace(1e55,1.1e55,5),alpha=0.3,label='Gamma-ray')

        #to show F1 and F2
        if settings==True:
            
            FO12=st.checkbox("show frequency for 1st and second ionisation of oxygen",value=True,key=p*100+1)
            if FO12:
                plt.plot([FO1,FO1],[0,10**50],label='First ionisation of oxygen')
                plt.plot([FO2,FO2],[0,10**50],label='second ionisation of oxygen')
    if spectrume :
        p=220202
        
        #show spectrum lines
        h1=h/(1.60217663e-19*1e+3)
        plt.fill_between(np.linspace(0,h1*3e9,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='radio')
        plt.fill_between(np.linspace(h1*3e9,h1*3e12,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='microwave')
        plt.fill_between(np.linspace(h1*3e12,h1*2.9999e14,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='infrared')
        plt.fill_between(np.linspace(h1*3.0001e14,h1*7.5e14,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='visible')
        plt.fill_between(np.linspace(h1*7.5e14,h1*3e16,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='UV')
        plt.fill_between(np.linspace(h1*3e16,h1*3e19,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='X-ray')
        plt.fill_between(np.linspace(h1*3e19,h1*3e34,5),np.linspace(1e52,1.1e52,5),alpha=0.3,label='Gamma-ray')

    if show_points:
        plt.scatter(x_list, y_list, marker='.', linestyle='-')
        plt.plot(x_list, y_list, marker='.', linestyle='-')

    else:
        plt.plot(x_list, y_list)

    plt.xscale('log')
    plt.yscale('log')
    
    if settings:   
        grid=st.checkbox('see grid', key="Grid_option")
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
        set_range=st.checkbox("set x and y range",key="Range_set",value=False)
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
