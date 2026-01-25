# ACCLUM — Accretion Disk Continuum and Luminosity Model

**ACCLUM** is an interactive, research-oriented software tool for computing and visualizing the spectral energy distribution (SED), temperature profile, and bolometric luminosity of standard thin accretion disks following the Shakura–Sunyaev formalism. The software is designed for astrophysics education, exploratory research, and generation of Cloudy-compatible spectral energy distributions.

The application is implemented as a Streamlit web interface with a modular scientific backend for disk physics, numerical integration, and spectral synthesis.

Live App:  
https://accretion-disk-spectrum.streamlit.app/

---

## Scientific Scope

ACCLUM implements the standard geometrically thin, optically thick accretion disk model and provides:

- Radial temperature profile \(T(r)\)
- Multi-temperature blackbody disk spectrum
- Frequency-dependent luminosity density \(L_\nu\)
- Bolometric luminosity via numerical integration
- Energy-space spectral representations
- Cloudy-compatible SED export
- Interactive exploration of disk parameters

The tool is intended for:

- Accretion disk physics studies
- AGN and black hole spectral modeling
- Teaching and visualization
- Preparing input continua for photoionization modeling (e.g., Cloudy)

---

## Physics Model

The effective temperature profile follows:

\[
T(r)^4 =
\left( \frac{3 G M_\bullet \dot{M}}{8 \pi \sigma} \right)
\left[ \frac{1 - \sqrt{r_i / r}}{r^3} \right]
\]

The monochromatic luminosity density is computed as:

\[
L_\nu =
\frac{16 \pi^2 h \nu^3}{c^2} \cos i
\int_{r_i}^{r_o}
\frac{r}{\exp\!\left(\frac{h\nu}{kT(r)}\right)-1} \, dr
\]

where:

- \(M_\bullet\) is the black hole mass  
- \(\dot{M}\) is the accretion rate  
- \(r_i, r_o\) are inner and outer disk radii  
- \(i\) is the inclination angle  
- \(G, h, k, c, \sigma\) have their usual meanings  

Numerical integration is performed using trapezoidal and Simpson’s rule methods.

---

## Software Architecture

The codebase is modularized to separate scientific logic from UI and I/O:
