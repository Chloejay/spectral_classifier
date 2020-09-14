## spectral_classifier

This repo is used to demo the feature engineering task for a data interview.

#### Data source 
Chemical Spectra and near-infrared spectroscopy data sets in Tobacco industry. 
- Infra-Red (IR), Near Infra-Red (NIR) 近红外光谱 and Raman Spectroscopy (拉曼散射光谱) theory =>特征峰的波数和强度
- Sepctral analysis in chemometrics field (化学计量学).


#### Experiments, typical for classification modeling
- Find peaks of spectrum/ PCA /KNN, for components of some samples are overlap, so PCA is not a good option
- SVM(after experiments and random checking, SVM has better performance)
- CNN for deep feature extraction, input: spectrogram(TODO)

after training I found my model is blind fool me, should return back to pre-processing of near-infrared (NIR) spectral data, for chemometrics modelling processing data is critical for model. 
**(for time limited, I only start this task from this Sunday afternoon, much research work and code refactor is under the progress in limited deadline.)**


###### Sample wavelength of Absorbance(AU).
<img src="wavelength.png" width="600" height="400">

#### Reference
- <a href="https://www.hindawi.com/journals/jamc/2020/9652470/">Classification Modeling Method for Near-Infrared Spectroscopy of Tobacco Based on Multimodal Convolution Neural Networks</a>
- <a href="https://www.researchgate.net/publication/226296679_A_Machine_Learning_Application_for_Classification_of_Chemical_Spectra">A Machine Learning Application for Classification of Chemical Spectra</a>
- <a href="https://arxiv.org/pdf/1707.08908.pdf">Deep Learning Models for Wireless Signal Classification with Distributed Low-Cost Spectrum Sensors</a>
- <a href="https://www.researchgate.net/publication/294138311_Model-based_pre-processing_in_Raman_spectroscopy_of_biological_samples">Model-based pre-processing in Raman spectroscopy of biological samples</a>
- <a href="https://wis.kuleuven.be/stat/robust/papers/2012/VerbovenHubertGoos-revision.pdf">Robust preprocessing and model selection for spectral data</a>