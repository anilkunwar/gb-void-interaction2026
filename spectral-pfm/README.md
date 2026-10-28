 # Phase Field FFT Solver (Allen-Cahn  and ElastoPlasticity Mechanics) for Simulation of Voids in UO2 Nuclear Fuel Material

 ## eta OP = void variable, phi OPs= grain variable
 ## r1-r2: Flexible domain size, t in s  ( Elastoplasticity : J2 Flow Rule)

[![continuummodelnt2d](https://img.shields.io/badge/voidgb1-streamlit-violet)](https://uo2-grain-boundary-void-evolution1.streamlit.app/) (modules and features: Non-physical Double-Well Shape in Free Energy, Non-physical grain separation )

[![continuummodelnt2d](https://img.shields.io/badge/voidgb2-streamlit-violet)](https://uo2-grain-boundary-void-evolution2.streamlit.app/) (modules and features: Non-physical Double-Well Shape in Free Energy, Non-physical grain separation )

## r3: Flexible domain size, t in s  ( Elastoplasticity : J2 Flow Rule, Free energy expression modified to prevent the separation of grains)

[![continuummodelnt2d](https://img.shields.io/badge/voidgb3-streamlit-violet)](https://uo2-grain-boundary-void-evolution3.streamlit.app/) (modules and features: (i) Symmetric double well potentials for void OP eta and grains OPs phi. The non-physical grain separation was fixed when (a)  the polynomial term in compute_dF_deta(),  was changed from eta^3 - eta to 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta) , and (b) the polynomial term in compute_dF_deta() was similarly changed from phi_i^3 - phi_i to 2.0 * phi_i * (1.0 - phi_i) * (1.0 - 2.0 * phi_i). The x^3 - x has an unstable maximum at x=0 and so the solution tends to be x --> 1. (ii) Reduced Gradient Energy Coefficient for Grain OPs, (iii) Narrowing initial Grain Boundary width from 2.5 nm to 1.0 nm. )

## r4: Flexible domain size, t in s  ( Elastoplasticity : J2 Flow Rule + Irradiation induced Creep, Free energy expression modified to prevent the separation of grains)

[![continuummodelnt2d](https://img.shields.io/badge/voidgb4-streamlit-violet)](https://uo2-grain-boundary-void-evolution4.streamlit.app/) (modules and features: (i) Symmetric double well potentials for void OP eta and grains OPs phi. The non-physical grain separation was fixed when (a)  the polynomial term in compute_dF_deta(),  was changed from eta^3 - eta to 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta) , and (b) the polynomial term in compute_dF_deta() was similarly changed from phi_i^3 - phi_i to 2.0 * phi_i * (1.0 - phi_i) * (1.0 - 2.0 * phi_i). The x^3 - x has an unstable maximum at x=0 and so the solution tends to be x --> 1. (ii) Reduced Gradient Energy Coefficient for Grain OPs, (iii) Narrowing initial Grain Boundary width from 2.5 nm to 1.0 nm. )
