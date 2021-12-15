# Hospital Allocation Planning using EHR data

### Overview
In this work, the goal is to develop a system that could help with the hospital allocation planning given patients' electronic health record (EHR) data. Patients' EHR data consists of tabular and textual information, and both of these information will be encoded and further process to estimate:
1. When the patient will be readmitted
2. What kind of disease that might occur on the following readmission
3. How severe is the disease might be on the following readmission

By estimating these 3 objectives from all patients, a hospital can anticipate unexpected load and refine their resource allocation more effectively. This solution is based on a hypothesis that EHR data can be used as a proxy to represent patients' health condition which can be helpful to approximate the potential disease and the severity of the disease within a certain period of time. To test this hypothesis, experiment on the EHR data is conducted. There are 4 experiments will be conducted:
1. An exploration to test whether **models built from a large-scale EHR data** can accurately estimate the patient readmission duration and potential disease risk with its severity-level given a patient health record data.
2. An exploration to test whether **a personalize models built from a patient specific health record data** can accurately estimate the patient readmission duration and potential disease risk with its severity-level
3. **An exploration of methods to combine (1) and (2)** to better estimate the patient readmission duration and potential disease risk with its severity-level
4. An experiment to test the developed methods with different patients' characteristic that has never been learnt by the model. This help to provide insights regarding **the generalization of the method** when it is directly/indirectly applied to the other hospital.

The result of these experiments will show the potential of utilizing a large-scale EHR data in a hospital for estimating a hospital's required allocation to obtain a more effective resource allocation in a hospital. For better interaction with the end-user, a dashboard showing estimated load will also be developed.

