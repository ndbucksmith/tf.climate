Using geotiff data to model climate with machine learning.

Tensorflow models are trained using climate data. The learned functions predict
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

Currently there are  three models

1. A simple neaural netowrk using annual averages for up to 17 features
2. A simple linear model, originally using only surface solar power and elevation, originally called stupidModel.  However it outperforms the neural net, especially after adding toa power ratio, i.e. surface solar power / toa power.  This ratio is a pretty good measure of greenhouse power magnitude. Given its [respectable] performance it has been renamed artisinal model ;)
3. A recurrent neural network that takes montly values for radiation at surface and toa plus precipitation and wind plus the annual averages.  



### Data Sources 

https://globalsolaratlas.info/downloads/world

http://worldclim.org/version2

Fick, S.E. and R.J. Hijmans, 2017. Worldclim 2: New 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology.


