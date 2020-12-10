## PedSimCity-Evaluation -  A workflow to evaluate the output of PedSimCity
 
This repository includes the necessary data and code to reproduce the analysis reported in *Modelling the effect of landmarks on pedestrian dynamics in urban environments*, published in *Computers, Environment and Urban Systems*.

The notebook *Modelling_Landmarks_Evaluation - London* can be used to evaluate the output of an ABM for pedestrian movement simulation, wherein landmarks are incorporated in the pedestrian agents' route choice models.
Such an ABM and the necessary input data are available [here](https://github.com/g-filomena/pedSimCity/tree/LandmarkBased).
Please note that the figures in the original paper were rendered in QGis, using the outputs of the jupyter notebook.

The notebooks *Urban_Subdivision_Evaluation - London* and *Urban_Subdivision_Evaluation - Paris* can be used to evaluate the output of an ABM for pedestrian movement simulation, wherein regions and barriers are incorporated in the pedestrian agents' route choice models.
Such an ABM and the necessary input data are available [here](https://github.com/g-filomena/pedSimCity/tree/RegionBased). 

The notebook *Urban_Subdivision_Evaluation - Plots* generates the figures presented in the paper *Perception of urban subdivisions in pedestrian movement simulation*, published in *PLOS ONE*.

To run the jupyter notebooks you need to install the library [cityImage](https://github.com/g-filomena/cityImage) and its dependicies.


