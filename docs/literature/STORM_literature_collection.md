# STORM-PhysNet — Literature Collection (compiled 10 Oct 2026)

Annotated literature for the STORM-PhysNet paper and its extended version.
Legend: V = verified (publisher/Crossref/Semantic Scholar), PV = partly verified, FLAG = check before citing.

## 1. Positioning against closest prior work
- **Closest prior art:** Sun et al. 2024 (JSWSC 14:25) — Transformer vs LSTM for ≥2 MeV GEO flux at 1, 3, 6, 12, 24 h; PE 0.940/0.886/0.828/0.747/0.660 (GOES-10, test 2005). Persistence for 1-h fluence 0.955 vs Transformer 0.961. Gap-filled by interpolation; inputs chosen on test year.
- Sun et al. 2025 — pre-training + few-shot, GOES-15 daily fluence PE 0.885→0.925 (transfer-like, single satellite).
- Zhao et al. 2025 (Space Weather 23:e2025SW004419) — cross-calibrated FY-4A + GOES-15, merged training beats GOES-15-only (calibration + merging, not fine-tuning).
- Tan et al. 2024 — Transformer vs TPA-LSTM difference **not statistically significant** (>0.8 MeV).
- Kieokaew et al. 2026 (arXiv:2605.15752) — Transformer encoder + TimesFM at 6 h; GOES-15→17 linear rescaling.
- Zeng et al. 2023 — linear models can beat Transformers → need simple baselines.
- Gaps we can own: residual-over-persistence heads; skill vs persistence reported per horizon; β-NLL + asymmetric storm loss; multi-seed paired tests; leakage-free evaluation; blocked test periods; GOES-15→GRASP (no peer-reviewed GRASP paper found).
- Caveat: OMNI is already time-shifted to the bow shock (King & Papitashvili 2005), so the learnable τ models a residual delay, not L1 transit.
- TFT (Lim 2021) already uses gating; the STORM gate is distinguished by its Bz conditioning.

## 2. Best reported numbers (not directly comparable — different satellites/years/cadence)
| Horizon | Best | Source |
|---|---|---|
| 1 h | PE 0.97, RMSE 0.18 | Son 2022 (MLP) |
| 1 h | PE 0.940 | Sun 2024 (GOES-10, 2005) |
| 6 h | PE 0.828 | Sun 2024 |
| 12 h | PE 0.747 | Sun 2024 |
| 24 h (5-min) | PE 0.660 | Sun 2024 |
| 24 h daily fluence | PE 0.925 | Sun 2025 |
| 24 h daily | PE 0.77 operational / 0.819 past-cast | SNB3GEO, Balikhin 2016 |
| 72 h | PE 0.53 | Son 2022 |

## 3. IEEE reference list (core, verified)
[1] D. N. Baker, P. J. Erickson, J. F. Fennell, J. C. Foster, A. N. Jaynes, and P. T. Verronen, "Space weather effects in the Earth's radiation belts," Space Sci. Rev., vol. 214, no. 1, Art. no. 17, 2018, doi: 10.1007/s11214-017-0452-7.
[2] R. B. Horne et al., "Wave acceleration of electrons in the Van Allen radiation belts," Nature, vol. 437, pp. 227–230, 2005, doi: 10.1038/nature04033.
[3] D. N. Baker, R. L. McPherron, T. E. Cayton, and R. W. Klebesadel, "Linear prediction filter analysis of relativistic electron properties at 6.6 R_E," J. Geophys. Res., vol. 95, no. A9, pp. 15133–15140, 1990, doi: 10.1029/JA095iA09p15133.
[4] K. L. Perry, G. P. Ginet, A. G. Ling, and R. V. Hilmer, "Comparing geosynchronous relativistic electron prediction models," Space Weather, vol. 8, Art. no. S12002, 2010, doi: 10.1029/2010SW000581.
[5] A. G. Ling, G. P. Ginet, R. V. Hilmer, and K. L. Perry, "A neural network-based geosynchronous relativistic electron flux forecasting model," Space Weather, vol. 8, Art. no. S09003, 2010, doi: 10.1029/2010SW000576.
[6] M. A. Balikhin et al., "Comparative analysis of NOAA REFM and SNB3GEO tools for the forecast of the fluxes of high-energy electrons at GEO," Space Weather, vol. 14, pp. 22–31, 2016, doi: 10.1002/2015SW001303.
[7] L. Wei, Q. Zhong, R. Lin, J. Wang, S. Liu, and Y. Cao, "Quantitative prediction of high-energy electron integral flux at geostationary orbit based on deep learning," Space Weather, vol. 16, no. 7, pp. 903–916, 2018, doi: 10.1029/2018SW001829.
[8] J. Son, Y.-J. Moon, and S. Shin, "72-hour time series forecasting of hourly relativistic electron fluxes at geostationary orbit by deep learning," Space Weather, vol. 20, no. 10, Art. no. e2022SW003153, 2022, doi: 10.1029/2022SW003153.
[9] X. Sun et al., "A modeling study of ≥2 MeV electron fluxes in GEO at different prediction time scales based on LSTM and transformer networks," J. Space Weather Space Clim., vol. 14, Art. no. 25, 2024, doi: 10.1051/swsc/2024021.
[10] X. Sun, R. Lin, S. Liu, B. Luo, S. Liu, and X. Dong, "Enhancing ≥2 MeV electron fluence predictions in GEO orbit through three deep learning strategies," Space Weather, vol. 23, no. 8, Art. no. e2025SW004519, 2025, doi: 10.1029/2025SW004519.
[11] X. Sun et al., "Prediction models of ≥2 MeV electron daily fluences for 3 days at GEO orbit using a long short-term memory network," Remote Sens., vol. 15, no. 10, Art. no. 2538, 2023, doi: 10.3390/rs15102538.
[12] M. Tan, X. Si, S. Teng, X. Wu, and X. Tao, "Comparative analysis of TPA-LSTM and Transformer models for forecasting GEO radiation belt electron fluxes," Space Weather, vol. 22, no. 11, Art. no. e2024SW004119, 2024, doi: 10.1029/2024SW004119.
[13] H. Zhang et al., "A prediction model of relativistic electrons at geostationary orbit using the EMD-LSTM network and geomagnetic indices," Space Weather, vol. 20, no. 10, Art. no. e2022SW003126, 2022, doi: 10.1029/2022SW003126.
[14] H. Zhang, S. Fu, L. Xie, D. Zhao, and C. Yue, "Relativistic electron flux prediction at geosynchronous orbit based on the neural network and the quantile regression method," Space Weather, vol. 18, Art. no. e2020SW002445, 2020, doi: 10.1029/2020SW002445.
[15] C. Forsyth, C. E. J. Watt, M. K. Mooney, I. J. Rae, S. D. Walton, and R. B. Horne, "Forecasting GOES 15 >2 MeV electron fluxes from solar wind data and geomagnetic indices," Space Weather, vol. 18, no. 8, Art. no. e2019SW002416, 2020, doi: 10.1029/2019SW002416.
[16] S. Sinha, Y. Chen, Y. Lin, and R. Pires de Lima, "PreMevE update: Forecasting ultra-relativistic electrons inside Earth's outer radiation belt," Space Weather, vol. 19, no. 9, Art. no. e2021SW002773, 2021, doi: 10.1029/2021SW002773.
[17] X. Chu et al., "Relativistic electron model in the outer radiation belt using a neural network approach," Space Weather, vol. 19, no. 12, Art. no. e2021SW002808, 2021, doi: 10.1029/2021SW002808.
[18] R. Kieokaew, F. Ginisty, R. Guezzi, and H. Mariaccia, "Forecasting megaelectron-volt electron flux in the Earth's outer radiation belt using supervised machine learning algorithms and time series foundation model," arXiv:2605.15752, 2026.
[19] E. Camporeale, "The challenge of machine learning in space weather: Nowcasting and forecasting," Space Weather, vol. 17, no. 8, pp. 1166–1207, 2019, doi: 10.1029/2018SW002061.
[20] A. Vaswani et al., "Attention is all you need," in Proc. NeurIPS, 2017, pp. 5998–6008.
[21] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are Transformers effective for time series forecasting?," in Proc. AAAI, vol. 37, no. 9, 2023, pp. 11121–11128, doi: 10.1609/aaai.v37i9.26317.
[22] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," Int. J. Forecast., vol. 37, no. 4, pp. 1748–1764, 2021, doi: 10.1016/j.ijforecast.2021.03.012.
[23] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with Transformers," in Proc. ICLR, 2023.
[24] J. H. King and N. E. Papitashvili, "Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data," J. Geophys. Res., vol. 110, Art. no. A02104, 2005, doi: 10.1029/2004JA010649.
[25] J. Matzka, C. Stolle, Y. Yamazaki, O. Bronkalla, and A. Morschhauser, "The geomagnetic Kp index and derived indices of geomagnetic activity," Space Weather, vol. 19, no. 5, Art. no. e2020SW002641, 2021, doi: 10.1029/2020SW002641.
[26] S. K. Morley, T. V. Brito, and D. T. Welling, "Measures of model performance based on the log accuracy ratio," Space Weather, vol. 16, no. 1, pp. 69–88, 2018, doi: 10.1002/2017SW001669.
[27] M. W. Liemohn et al., "Model evaluation guidelines for geomagnetic index predictions," Space Weather, vol. 16, no. 12, pp. 2079–2102, 2018. (FLAG DOI)
[28] M. Seitzer, A. Tavakoli, D. Antic, and G. Martius, "On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks," in Proc. ICLR, 2022.
[29] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Proc. NeurIPS, 2017, pp. 6402–6413.
[30] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, Art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804.
[31] C. Bergmeir, R. J. Hyndman, and B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," Comput. Stat. Data Anal., vol. 120, pp. 70–83, 2018, doi: 10.1016/j.csda.2017.11.003.
[32] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," Ecography, vol. 40, no. 8, pp. 913–929, 2017, doi: 10.1111/ecog.02881.
[33] Y. Zhao et al., "Cross-calibrated FY-4A and GOES-15 data for >2 MeV electron flux prediction at geosynchronous orbit: A machine learning approach," Space Weather, vol. 23, no. 11, Art. no. e2025SW004419, 2025, doi: 10.1029/2025SW004419.
[34] X. Y. Wang et al., "Cross-satellite calibration of high-energy electron fluxes measured by FengYun-4A based on Arase observations," Earth Planet. Phys., vol. 7, no. 5, pp. 565–575, 2023, doi: 10.26464/epp2023076.
[35] Indian Space Science Data Centre, "GRASP — Geostationary RAdiation SPectrometer," PRADAN. [Online]. Available: https://pradan1.issdc.gov.in/grasp/

## 4. Additional entries (for IEEE Access version / extra context)
Physics & hazards: Meredith et al. 2015 Space Weather 13:170–184 doi:10.1002/2014SW001143 (extreme GOES >2 MeV fluxes, PV); Han et al. 2024 Solar Syst. Res. doi:10.1134/S0038094624700357 (review, PV); ICARE_NG2 2025 JSWSC (FLAG).
Classical/operational: Boynton et al. 2013 JGR 118(4):1500–1513 NARMAX (FLAG DOI); Boynton et al. 2015 Ann. Geophys. online NARMAX (FLAG); Boynton et al. 2016 Space Weather doi:10.1002/2016SW001506 (PV); Walker et al. 2022 Space Weather e2022SW003124 (PV); SNB3GEO LANL/GOES-13 evaluation 2022 Adv. Space Res. (FLAG); Turner et al. 2011 Space Weather doi:10.1029/2010SW000647 (PV); Sakaguchi et al. 2015 Space Weather MVAR (FLAG DOI); Myagkova et al. 2017 Geomagn. Aeron. 57 doi:10.1134/S0016793217010108 (1–12 h, NN ensemble, PV); Medium-term fluence prediction 2022 Adv. Space Res. (FLAG); Bernoux et al. 2021 JSWSC 11 (FLAG).
ML/DL: Sun et al. 2021 Remote Sens. 13(17):3347 doi:10.3390/rs13173347 (longitude mapping); Qian et al. 2020 Space Weather 18(8) doi:10.1029/2018SW002078 (PV); Tang et al. 2022 Space Weather e2021SW002969 (stacking ensemble, FLAG authors); Landis et al. 2022 Space Weather e2021SW002774 NARX (FLAG DOI); Katsavrias et al. 2022 Space Weather e2021SW002936 (FLAG title); Ma et al. 2022 Space Weather e2022SW003079; Ma et al. 2023 Space Weather e2022SW003339 (DeepSHAP); Pires de Lima et al. 2020 PreMevE 2.0 (FLAG); Feng et al. 2024 PreMevE-MEO e2024SW003975 (PV); Boyd et al. 2023 SHELLS e2022SW003338 (PV); Botek et al. 2023 PROBA-V e2023SW003466 (PV); Moon et al. 2023 Proc. IAU doi:10.1017/S1743921322004847; ANN 8–10-day lags 2022 Adv. Space Res. (FLAG).
Transformers/TS: Zhou 2021 Informer (AAAI 35(12):11106–11115); Wu 2021 Autoformer (NeurIPS); Zhou 2022 FEDformer (ICML, PMLR 162); Liu 2024 iTransformer (ICLR); Zhang & Yan 2023 Crossformer (ICLR); Kim 2022 RevIN (ICLR); Chen 2023 TSMixer (TMLR); Das 2023 TiDE arXiv:2304.08424; Upendran 2022 Space Weather e2022SW003045; Liu 2020 TEC Space Weather e2020SW002501; IonCast 2025 arXiv:2511.15004 (FLAG); Laperre 2020 arXiv:2006.04667 (Dst DTW — persistence-like forecasts); AE turbulence 2026 arXiv:2606.16518 (FLAG); Hochreiter & Schmidhuber 1997 LSTM.
Physics-informed / propagation: Al Shidi 2024 Space Weather e2023SW003747; McGranaghan 2021 arXiv:2011.10117.
Metrics/uncertainty: Liemohn 2021 JASTP 218:105624 ("RMSE is not enough"); Kendall & Gal 2017 NeurIPS.
Validation: Bergmeir & Benítez 2012 Inf. Sci. 191:192–213 doi:10.1016/j.ins.2011.12.028.
Cross-satellite: Liu et al. 2019 FY-4A/GOES-13 cross-calibration (FLAG); Oh et al. 2024 Earth Planets Space 76:46 doi:10.1186/s40623-024-01992-y (KSEM vs GOES-16); GPS electron cross-calibration 2026 Satellite Navigation doi:10.1186/s43020-026-00203-1 (FLAG); Rodriguez et al. 2020 JSWSC 10 (FLAG); Bruno 2017 arXiv:1708.05641.
Data: OMNI = King & Papitashvili 2005; Kp = Matzka 2021; GOES via NOAA NCEI; Dst/SYM-H via WDC Kyoto (cite with access dates). GRASP on GSAT-19 per ISSDC.

## 5. Likely paywalled
Horne 2005 (Nature); Han 2024 (Solar Syst. Res.); Myagkova 2017 (Geomagn. Aeron.); three Adv. Space Res. 2022 papers (SNB3GEO evaluation; medium-term fluence; ANN 8–10-day lags); Bergmeir & Benítez 2012; Bergmeir 2018; Lim 2021 (arXiv copy exists); Roberts 2017. Older AGU papers usually free after login.

## 6. Still unverified (not in list)
NOAA REFM docs, Reeves 2011, VERB/DREAM/Salammbô, Weimer propagation, PINN reviews, TimesNet, GOES-15 EPEAD/MAGED and GOES-R SEISS instrument papers, WDC Kyoto indices.
