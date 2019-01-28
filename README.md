Using geotiff data to model climate with machine learning.

A tensorflow model is trained using climate data. The learned function predicts
average temperature at any location as a function of:

* vis radiation down
* TOA vis radiation (pure function of latitude)
* elevation
* barometric pressure as a function of elevation
* precipitation
* land or water or ice one hot
* hemisphere onehot
* albedo
* convolutions of surrouning area such as
   * NS and EW slope
   * STD of elevation


The  goal of this project is a validated differentiable function that can be used to determine climate sensitivity to greenhouse gas warming.



# Data Sources 

https://globalsolaratlas.info/downloads/world

http://worldclim.org/version2

Fick, S.E. and R.J. Hijmans, 2017. Worldclim 2: New 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology.


