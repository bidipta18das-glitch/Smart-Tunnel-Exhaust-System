<div align="center">

<img src="https://www.sju.edu.in/assets/img/st-joseph-university-logo.png" height="80" style="background:white; padding:8px; margin:0 16px;" />
<img src="https://www.erafoundationindia.org/images/logo.svg" height="80" style="background:white; padding:8px; margin:0 16px;" />
<img src="https://comedkares.org/wp-content/uploads/2023/04/Comedkares-Logo-EPS.png" height="80" style="background:white; padding:8px; margin:0 16px;" />

</div>

---

# Tunnel Exhaust System Project -ML model for predicting air quality in tunnels

<div align="center">

**Bidipta Das**, Dept of Computer Science, 252CSC23 &nbsp;·&nbsp; **Amala Sharon**, Dept of Advanced Computing, 252BDA50 &nbsp;·&nbsp;**Joe Vinny Rozario**, Dept of Advanced Computing, 252BDA53 &nbsp;·&nbsp; **Francis Aakash**, Dept of Advanced Computing, 252BDA35

</div>

---

## Abstract

## Road tunnels often experience the accumulation of smoke, carbon monoxide, and hazardous gases due to vehicle emissions and accidents. Poor air quality inside tunnels can reduce visibility and pose serious health risks to commuters. This paper presents an IoT-based tunnel ventilation system that continuously monitors environmental conditions using gas, smoke, temperature, and humidity sensors. The collected data is processed by an ESP8266 microcontroller, which automatically controls exhaust fans based on detected pollution levels. The system enables real-time monitoring and improves tunnel safety by ensuring proper air circulation and rapid removal of harmful pollutants. Experimental results demonstrate that the proposed system effectively responds to varying environmental conditions and enhances air quality management in road tunnels.

## Keywords

Tunnel Ventilation System, Internet of Things (IoT), ESP8266, Air Quality Monitoring, Smoke Detection, Gas Detection, DHT11 Sensor, Automated Exhaust Control, Machine Learning, Data Analysis, Environmental Monitoring, Smart Transportation Infrastructure

---

# I. Introduction

Tunnel roads have become an essential part of modern transportation infrastructure, helping to reduce traffic congestion and improve connectivity in urban and mountainous regions. However, tunnels are enclosed environments where vehicular emissions such as carbon monoxide (CO), smoke, particulate matter, and other harmful pollutants can accumulate rapidly. These pollutants not only reduce air quality but also create serious health and safety risks for commuters. In addition, tunnel accidents and fire incidents can lead to the generation of large amounts of toxic smoke, making evacuation difficult and increasing the risk of casualties. Therefore, efficient ventilation systems are critical for maintaining safe operating conditions inside road tunnels [1].

Previous studies have highlighted various challenges associated with tunnel environments. Król and Król investigated smoke movement and hot gas flow during tunnel fire incidents and demonstrated the importance of properly functioning ventilation systems for removing smoke and ensuring safe evacuation during emergencies [2]. Their work showed that ventilation systems are commonly activated when pollutant concentrations exceed permissible limits or when visibility decreases. Similarly, Kim proposed a deep-learning-based vehicle detection system for tunnel environments and emphasized that tunnels present unique challenges due to poor lighting conditions, exhaust gas pollution, and reduced visibility, all of which contribute to increased accident risks [3]. While these studies contribute significantly to tunnel safety, they primarily focus on fire behavior analysis and vehicle detection rather than intelligent environmental monitoring and adaptive ventilation control.

Despite the availability of conventional ventilation systems, many existing solutions operate continuously or rely on fixed threshold mechanisms, resulting in excessive energy consumption and limited adaptability to changing environmental conditions. Furthermore, most tunnel monitoring systems focus either on environmental sensing or safety analysis without integrating real-time hardware control and predictive data analysis. This creates a need for a cost-effective and intelligent ventilation system capable of monitoring tunnel conditions, responding automatically to hazardous situations, and utilizing collected data for future predictive analysis.

To address these challenges, this paper proposes an IoT-based smart tunnel ventilation system that integrates MQ-2 smoke sensors, MQ-3 gas sensors, DHT11 temperature and humidity sensors, an ESP8266 microcontroller, relay-based fan control, and machine learning-based data analysis. The proposed system continuously monitors environmental conditions inside the tunnel and automatically activates exhaust fans when hazardous conditions are detected. In addition, the collected sensor data can be analyzed using machine learning techniques to identify pollution patterns and support future predictive ventilation strategies. The proposed approach aims to improve tunnel safety, enhance air quality management, and reduce unnecessary energy consumption through intelligent and automated control.

The main contributions of this work are: (i) the development of a low-cost IoT-based tunnel monitoring platform, (ii) the implementation of automated exhaust fan control using real-time sensor data, (iii) the integration of environmental monitoring with machine learning-based analysis, and (iv) the demonstration of a scalable framework for smart tunnel infrastructure. The remainder of this paper is organized as follows: Section II reviews related work, Section V presents the proposed methodology and system architecture, Section VI discusses implementation and experimental results, and Section X concludes the paper with future research directions.

---

## II. Literature Review

Tunnel fire safety has been an important area of research due to the confined nature of tunnels and the rapid spread of smoke and toxic gases during fire incidents. Early studies focused on understanding fire dynamics and smoke propagation within tunnels. Król and Król investigated hot gas flow and temperature distribution in a real road tunnel using full-scale fire experiments. Their findings highlighted the importance of efficient ventilation systems and rapid fire detection for safe evacuation and smoke control during emergencies [1].

As sensor technologies evolved, researchers explored multi-sensor approaches to improve fire detection reliability. Wang et al. proposed an improved Dempster–Shafer (DS) evidence theory framework that fused temperature, smoke concentration, and CO sensor data to address uncertainty and conflicting evidence in tunnel fire monitoring. Their two-level data fusion strategy improved fire detection probability and reduced response time compared with conventional methods [2].

Recent advancements in deep learning have enabled more accurate and real-time fire detection systems. Zheng et al. developed an improved YOLOv5s-based algorithm specifically for tunnel fire detection. By optimizing the backbone structure, introducing a Coordinate Attention (CA) mechanism, and replacing the traditional loss function with Wise-IoU, the proposed model achieved high detection accuracy while maintaining real-time performance. Experimental results demonstrated a precision of 98.9% and a detection speed of 148 FPS, making the system suitable for practical tunnel monitoring applications [3].

Further improvements have been achieved through the integration of edge computing technologies. Li et al. proposed an intelligent tunnel fire smoke monitoring framework based on an enhanced YOLOX architecture deployed on edge devices. The study incorporated wavelet-transform-based feature extraction, knowledge distillation, and model quantization to reduce computational requirements while maintaining detection accuracy. The proposed approach achieved higher accuracy than the baseline YOLOX model and demonstrated the feasibility of real-time tunnel fire monitoring on resource-constrained edge platforms [4].

Collectively, these studies illustrate the evolution of tunnel fire detection from traditional fire behavior analysis and ventilation studies [1] to intelligent multi-sensor fusion techniques [2] and advanced deep learning-based vision systems [3], [4]. While deep learning approaches provide superior accuracy and real-time detection capabilities, integrating them with multi-sensor fusion frameworks may further enhance reliability and robustness in complex tunnel environments.

---

## III. Problem Statement

Despite the availability of ventilation systems in enclosed infrastructure, three key challenges have limited the practical deployment of automated exhaust control in tunnel environments:

i. Delayed human response. Conventional tunnel ventilation systems rely on manual monitoring or fixed timer-based operation, which cannot respond quickly enough to sudden changes in temperature or gas concentration. In emergency scenarios such as fires or gas leaks, delayed response can lead to serious safety hazards for personnel and equipment inside the tunnel.

ii. Lack of real-time environmental sensing. Many existing low-cost ventilation setups operate without continuous sensor feedback. Without real-time data on temperature, humidity, and smoke levels, the system has no way to distinguish between safe and unsafe conditions, leading to either over-ventilation or dangerous under-ventilation.

iii. High cost and complexity of existing solutions. Industrial-grade tunnel monitoring systems typically require expensive hardware, complex wiring infrastructure, and specialized installation. This places automated environmental control beyond the reach of small-scale or educational tunnel setups where safety monitoring is equally important.

This work addresses all three challenges: it establishes that a low-cost NodeMCU ESP8266 microcontroller paired with a DHT11 temperature sensor and MQ-2 smoke sensor is sufficient for real-time environmental monitoring, demonstrates threshold-based automatic fan control as a reliable approach to immediate hazard response, and validates the system through live serial monitoring to confirm sensor accuracy and actuation correctness

---

## 4. Objectives

The primary objective of this project is to design and implement a low-cost automated tunnel exhaust system capable of detecting hazardous environmental conditions and responding without human intervention. The specific objectives are as follows:

i. To monitor real-time environmental conditions inside a tunnel environment using a DHT11 temperature and humidity sensor and an MQ-2 smoke and gas sensor connected to a NodeMCU ESP8266 microcontroller, ensuring continuous acquisition of temperature and smoke concentration data at regular intervals.

ii. To establish safe threshold values for temperature and smoke concentration beyond which the tunnel environment is considered hazardous, and to implement a threshold-based decision logic that evaluates incoming sensor data and determines whether ventilation is required.

iii. To implement automatic fan control using a TIP122 Darlington transistor circuit driven by the NodeMCU, such that the exhaust fan activates immediately when temperature exceeds 30°C or smoke concentration exceeds a defined analog threshold, and deactivates automatically when conditions return to safe levels.

iv. To validate the system through real-time serial monitoring, confirming that sensor readings are accurate, that the fan responds correctly to both temperature and smoke triggers, and that the system recovers to its normal state once hazardous conditions subside.

v. To demonstrate that low-cost, microcontroller-based hardware is sufficient for automated environmental monitoring and actuation in tunnel-like enclosed spaces, providing a practical and accessible alternative to expensive industrial ventilation control systems.

---

## 5. Methodology

### 5.1 System Architecture

The proposed tunnel exhaust system is designed as an automated environmental monitoring and ventilation framework that continuously observes tunnel conditions and activates ventilation when hazardous conditions are detected. The architecture consists of three primary layers: the sensing layer, the processing layer, and the actuation layer. The sensing layer acquires real-time environmental data through temperature, humidity, and gas sensors. The processing layer evaluates the sensor readings against predefined safety thresholds and determines the operational state of the tunnel. The actuation layer controls the exhaust fan based on the decisions generated by the processing layer. This layered architecture enables real-time monitoring and rapid response to unsafe environmental conditions.

### 5.2. Hardware Configuration

The hardware platform is centered around the NodeMCU ESP8266 microcontroller, which serves as the main processing and communication unit. Environmental data are collected using a DHT11 sensor for temperature and humidity measurement and an MQ-2 gas sensor for smoke and combustible gas detection. The exhaust fan is driven through a TIP122 Darlington transistor that acts as a switching device between the microcontroller and the external power source. The fan is powered using an 11.1 V external supply to ensure sufficient airflow for tunnel ventilation. This configuration provides a cost-effective and energy-efficient solution for automated tunnel exhaust control.

### 5.3. Environmental Data Acquisition

The NodeMCU continuously collects environmental measurements at regular intervals of three seconds. Temperature and humidity values are obtained from the DHT11 sensor connected to digital pin D4, while smoke concentration values are acquired from the MQ-2 sensor connected to the analog input pin A0. The sensor readings are transmitted to the processing module for evaluation. Continuous monitoring ensures that environmental changes inside the tunnel are detected promptly, allowing timely activation of the ventilation system when necessary.

### 5.4. Threshold-Based Safety Analysis

To determine the safety status of the tunnel environment, the acquired sensor readings are compared against predefined threshold values. The temperature threshold is set to 30°C, while the smoke threshold is fixed at a sensor value of 400 on a scale ranging from 0 to 1023. If either the temperature or smoke reading exceeds its corresponding threshold, the system categorizes the environment as potentially hazardous. This threshold-based approach provides a simple yet reliable mechanism for detecting unsafe tunnel conditions and initiating corrective actions.

### 5.5. Exhaust Fan Control Mechanism

The exhaust fan is controlled through a TIP122 Darlington transistor, which functions as an electronic switch. Since the NodeMCU operates at a logic voltage of 3.3 V and cannot directly drive the fan, the transistor provides the necessary current amplification. A digital HIGH signal from pin D1 is applied to the transistor base through a 1 kΩ resistor, enabling current flow from the external power source to the exhaust fan. When activated, the fan removes hot air and smoke from the tunnel environment, thereby improving air quality and reducing safety risks.

### 5.6. Automatic Ventilation Decision Logic

The control logic employed in the proposed system is based on real-time environmental conditions. The exhaust fan is activated whenever the measured temperature exceeds 30°C or the smoke concentration surpasses the threshold value of 400. If both parameters remain within the safe operating range, the fan remains deactivated. The decision-making process is repeated every three seconds, ensuring continuous assessment of tunnel conditions. This automated control strategy eliminates the need for manual intervention and enables rapid responses to hazardous situations.

### 5.7. Real-Time Monitoring and System Evaluation

The performance of the proposed system is evaluated through continuous monitoring of sensor readings and fan status. Real-time data collected from the DHT11 and MQ-2 sensors are displayed through a monitoring interface, allowing operators to observe environmental conditions within the tunnel. The effectiveness of the system is assessed by analyzing its ability to detect hazardous conditions and activate the ventilation mechanism promptly. Experimental observations demonstrate that the system successfully responds to elevated temperature and smoke levels, thereby maintaining safer tunnel operating conditions and improving environmental management.

---

## 6. Implementation

### 6.1. Hardware System Integration

The proposed tunnel exhaust system was implemented using a NodeMCU ESP8266 microcontroller as the central processing and control unit. Environmental monitoring was achieved through the integration of a DHT11 temperature and humidity sensor and an MQ-2 gas sensor. The DHT11 sensor was connected to digital pin D4 of the NodeMCU, while the MQ-2 sensor was interfaced through the analog input pin A0. An exhaust fan was connected through a TIP122 Darlington transistor, which functioned as a switching device between the microcontroller and the external 11.1 V power supply. The complete hardware setup was assembled on a prototype board and tested under various environmental conditions.

The experimental configuration is schematically illustrated in Fig. 1.

> **Fig. 1.** (a) Complete hardware architecture of the proposed tunnel exhaust system showing the NodeMCU, DHT11 sensor, MQ-2 sensor, TIP122 transistor, and exhaust fan connections.

### 6.2. Sensor Data Acquisition and Processing

The NodeMCU was programmed using the Arduino IDE to continuously acquire sensor readings at intervals of three seconds. Temperature and humidity values were collected from the DHT11 sensor, while smoke concentration values were obtained from the MQ-2 gas sensor. The acquired sensor readings were processed in real time and compared against predefined safety thresholds. To ensure reliable operation, invalid sensor readings were discarded and only validated measurements were considered for decision making.

The collected environmental data formed the basis for determining the operational state of the tunnel and the corresponding ventilation requirements.

### 6.3. Automatic Fan Control Implementation

A threshold-based control algorithm was implemented within the NodeMCU firmware. The temperature threshold was configured at 30°C and the smoke threshold at a sensor value of 400. Whenever either threshold was exceeded, the NodeMCU generated a HIGH signal at pin D1, activating the TIP122 transistor and switching ON the exhaust fan.

Conversely, when both temperature and smoke levels remained below their respective thresholds, the microcontroller generated a LOW signal, deactivating the transistor and turning the fan OFF. This control mechanism provided immediate ventilation whenever hazardous conditions were detected inside the tunnel.

### 6.4. Real-Time Communication and Monitoring

To enable remote monitoring, the NodeMCU transmitted sensor readings through the MQTT communication protocol over a Wi-Fi network. An MQTT broker handled the message exchange between the hardware system and the monitoring platform. Sensor values including temperature, humidity, smoke concentration, and fan status were published periodically to dedicated MQTT topics.

Node-RED was employed to subscribe to these topics and process incoming data streams. The platform enabled real-time visualization of tunnel conditions and provided an intuitive interface for monitoring system performance.

> **Fig. 2.** Real-time monitoring dashboard displaying temperature, humidity, smoke concentration, and exhaust fan status.

### 6.5. Dashboard Visualization and System Testing

A web-based dashboard was developed using Node-RED to visualize environmental conditions inside the tunnel. Gauge widgets, charts, and status indicators were utilized to display real-time sensor readings and fan activity. Historical sensor data were plotted to observe environmental trends and evaluate system behavior over time.

The implemented system was tested under various operating conditions by introducing heat and smoke near the sensors. Experimental observations confirmed that the system successfully detected unsafe environmental conditions and activated the exhaust fan automatically. The response time was found to be sufficient for practical tunnel ventilation applications, demonstrating the effectiveness of the proposed implementation.

---

## 7. Results & Analysis

### 7.1. Hologram Recording Quality

The exposure adjustment protocol successfully maintained the photoresist within its linear response regime across the hologram area. The developed photoresist relief patterns exhibited high-contrast holographic fringes consistent with good spatial coherence of the EUV illumination. The 100 nm aluminum foil test object transmitted approximately 35% of the 46.9 nm illumination [19] while effectively blocking lower-energy plasma background emission, yielding clean holographic fringes with high signal-to-noise ratio.

### 7.2. Numerical Reconstruction

The Fresnel propagation code successfully reconstructed images from both digitized holograms. Reconstructed amplitude images at the optimal back-propagation distance showed clear rendition of the test object features, including the spherical markers and the tilted mylar surface profile.

**Table II: Possible Ω Functions in Reconstruction**

| Range | Ω(m)             |
| ----- | ---------------- |
| x < 0 | Ω(m) = Σᵢ₌₀ᵐ K⁻ⁱ |
| x ≥ 0 | Ω(m) = √m        |

**Table III: Network and Reconstruction Delay as a Function of Load**

| β   | λ_min | λ_max   |
| --- | ----- | ------- |
| 1   | 0.057 | 0.172   |
| 10  | 0.124 | 0.536   |
| 100 | 0.830 | 0.905\* |

_\* limited usability_

### 7.3. Depth Resolution

Through numerical optical sectioning — sweeping the propagation distance across the full depth extent of the test object — the surface topography of the tilted test object was mapped with a depth resolution of approximately 2 μm. This resolution is governed primarily by the numerical aperture of the holographic recording: higher NA recordings provide finer depth discrimination. The tilted transparent surface with 465 nm spherical markers served as a calibrated reference for verifying that different depth planes were correctly separated in the sectioned image stack.

$$x = \sum_{l=0}^{z} 2^l Q \tag{3}$$

### 7.4. Lateral Resolution

Lateral resolution was assessed via wavelet image decomposition and image correlation analysis applied to the numerically reconstructed images. The best lateral resolution achieved, obtained from the highest NA recording, was **164 nm** — exceeding previously published results by more than a factor of two. This represents a significant advance for compact table-top EUV holographic imaging systems.

---

## 8. Discussion

The results confirm that three-dimensional volumetric imaging by numerical optical sectioning is achievable with a compact table-top EUV laser, without access to synchrotron or large-facility sources. The 164 nm lateral resolution achieved here surpasses prior table-top demonstrations by more than a factor of two, driven by the combination of a high-coherence capillary laser source and a high-NA holographic recording geometry.

The depth resolution of approximately 2 μm is consistent with theoretical expectations for the numerical aperture employed, and will improve proportionally with larger NA recordings. The use of photoresist as the recording medium — rather than a real-time detector — introduces a processing latency (development, AFM digitization) but provides a high-resolution, high-fidelity hologram record without detector noise or pixel pitch limitations.

The numerical Fresnel propagation approach is computationally efficient and well-suited to the digitized AFM data. The sectioning technique is robust against noise in the digitized hologram, as confirmed by the clean separation of depth planes in the reconstructed image stack.

One important practical consideration is that the entire pipeline — laser, vacuum chamber, photoresist processing, AFM, and numerical reconstruction — must be carefully calibrated and aligned. The 10⁻⁵ Torr vacuum requirement and the sensitivity of EUV optics to contamination impose operational constraints, but these are manageable within a well-equipped laboratory environment.

The technique has natural extensions to biological and materials science imaging, where three-dimensional nanoscale structure at EUV-relevant length scales is of significant interest. The compact source format is particularly enabling for such applications.

---

## 9. Conclusion

We have demonstrated that three-dimensional holographic imaging in a volume can be obtained from a single high-NA hologram recorded with a compact table-top λ = 46.9 nm EUV laser. The numerical optical sectioning technique — implemented by sweeping the propagation distance in a Fresnel reconstruction code — produces a robust three-dimensional image of a purpose-fabricated test object.

Quantitative analysis using wavelet image decomposition and image correlation confirmed a best lateral resolution of **164 nm**, representing an improvement of more than a factor of two relative to previously published table-top EUV holography results. Surface topography mapping by numerical sectioning achieved a depth resolution close to **2 μm**, with this figure governed by the numerical aperture of the holographic recording.

These results establish a complete, end-to-end table-top EUV holographic volumetric imaging pipeline that is accessible to individual research groups, and validate numerical optical sectioning as a practical route to three-dimensional nanoscale imaging with compact coherent EUV sources.

---

## 10. Future Scope

Building on the results of this work, several directions for future investigation are identified:

**Higher numerical aperture recordings.** Increasing the NA of the holographic exposure will improve both lateral and depth resolution simultaneously. Exploration of the achievable NA limits for the capillary EUV laser geometry is warranted.

**Real-time or near-real-time recording media.** Replacing photoresist with EUV-sensitive CCD or CMOS detectors would eliminate the development and AFM digitization steps, enabling faster acquisition cycles and time-resolved 3D imaging applications.

**Biological and materials samples.** Extension of the technique to imaging of biological cells, nanostructured materials, and semiconductor devices would demonstrate practical utility beyond the proof-of-principle test object used here.

**Improved reconstruction algorithms.** Phase retrieval algorithms and iterative reconstruction techniques may further improve image quality and resolution beyond the limits imposed by the direct Fresnel back-propagation approach.

**Wavelength extension.** Application of the numerical sectioning technique at shorter EUV or soft X-ray wavelengths — using either HHG sources or next-generation compact discharge lasers — would push lateral resolution toward and potentially below 10 nm.

**Automated depth-plane segmentation.** Development of automated image-analysis algorithms for identifying and segmenting depth planes within the optical section stack would enhance the practical utility of the technique for routine three-dimensional characterization.

---

## Acknowledgements

The authors wish to thank the anonymous reviewers for their valuable suggestions. This research was sponsored by the National Science Foundation through the NSF ERC Center for Extreme Ultraviolet Science and Technology, NSF Award No. EEC-0310717.

---

## References

[1] Y. Wang, H. Zhang, J. Li, and X. Chen, “Multi-Sensor Data Fusion-Based Tunnel Fire Detection and Identification Method,” Sensors, vol. 24, no. 20, p. 6455, 2024.

[2] A. Król and M. Król, “Study on Hot Gases Flow in Case of Fire in a Road Tunnel,” Energies, vol. 11, no. 3, p. 290, 2018.

[3] J. Kim, “Vehicle Detection Using Deep Learning Technique in Tunnel Road Environments,” Symmetry, vol. 12, no. 12, p. 2012, 2020.

[4] H. Wang, Y. Shi, L. Chen, and X. Zhang, "A Tunnel Fire Detection Method Based on an Improved Dempster-Shafer Evidence Theory," Sensors, vol. 24, no. 6455, 2024.

[5] C. Li, B. Zhu, G. Chen, Q. Li, and Z. Xu, "Intelligent Monitoring of Tunnel Fire Smoke Based on Improved YOLOX and Edge Computing," Applied Sciences, vol. 15, no. 2127, 2025.
