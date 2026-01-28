import datetime

import numpy as np
from dateutil.relativedelta import relativedelta

# Set of numeric features.
_NUMERIC_THINGSET = {
    "birth_yr",
    "BMI",
    "Weight",
    "Total Cholesterol",
    "eGFR",
    "LDL",
    "HbA1C",
    "Triglycerides",
    "Waist circumference",
}

# Set of time-invariant features.
_INVAR_THINGSET = {"pseudoage", "race", "gender"}

# Medication-related features.
_MEDS_THINGSET = {
    "ACE (with combos)",
    "ARB (with combos)",
    "GLP-1",
    "Insulin",
    "Metformin (with combos)",
    "Statin (with combos)",
}

# Set of indicative features.
_DIAGS_THINGSET = {
    "Chronic Ischemic Heart Disease",
    "Hypertension",
    "Atrial fibrillation",
    "Dyslipidemia",
    "Asthma",
    "Ischemic stroke",
    "pancreatic cancer",
    "Cardiomyopathy",
    "Heart Failure",
    "breast cancer ",
    "Acute Kidney Failure",
    "T2D",
    "Thyroid cancer",
    "COPD",
    "Arrhythmias excluding atrial fibrillation",
    "Hypertensive heart disease",
    "Cholelithiasis",
    "Heart Disease affecting pregnancy",
    "Hemorrhagic stroke",
    "Acute Myocardial Infarction",
    "colorectal cancer",
    "Cholecystitis",
    "Proteinuria",
    "Acute Pancreatitis",
    "Secondary Hypertension",
    "ovarian cancer",
    "Acute Ischemic Heart Disease excluding MI",
    "multiple myeloma",
    "gallbladder cancer",
    "Cardiac Arrest",
    "stomach cancer",
    "hepatocellular carcinoma",
    "esophageal cancer",
    "kidney cancer",
    "meningioma",
    "Subsequent Myocardial Infarction",
    "Complication of Myocardial Infarction",
    "Low Back Pain",
    "Sleep Apnea",
    "Primary CKD stage 1-4",
    "Hypertensive heart and kidney disease",
    "CKD 5 and ESRD",
    "multiple myeloma",
    "endometrial cancer",
    "Obesity with impaired breathing",
}

# Set of indicative features of interest.
_OUTCOMES_OF_INTEREST = {
    "Acute Myocardial Infarction",
    "Subsequent Myocardial Infarction",
    "Complication of Myocardial Infarction",
    "Acute Ischemic Heart Disease excluding MI",
    "Cardiac Arrest",
    "Arrhythmias excluding atrial fibrillation",
    "Ischemic stroke",
    "Hemorrhagic stroke",
    "Atrial fibrillation",
    "Cardiomyopathy",
    "Cholelithiasis",
    "Cholecystitis",
    "Acute Pancreatitis",
    "esophageal cancer",
    "stomach cancer",
    "colorectal cancer",
    "hepatocellular carcinoma",
    "gallbladder cancer",
    "pancreatic cancer",
    "breast cancer ",
    "endometrial cancer",
    "ovarian cancer",
    "kidney cancer",
    "meningioma",
    "Thyroid cancer",
    "multiple myeloma",
    "CKD 5 and ESRD",
    "Primary CKD stage 1-4",
    "Acute Kidney Failure",
    "Sleep Apnea",
    "COPD",
    "patient_death",
}


def is_numeric(thing: str) -> bool:
    return thing in _NUMERIC_THINGSET


def is_categorical(thing: str) -> bool:
    return thing not in _NUMERIC_THINGSET


def is_medication(thing: str) -> bool:
    return thing in _MEDS_THINGSET


def is_diagnosis(thing: str) -> bool:
    return thing in _DIAGS_THINGSET


def is_outcome_of_interest(thing: str) -> bool:
    return thing in _OUTCOMES_OF_INTEREST


def is_sorted(a: list) -> np.bool_:
    return np.all(a[:-1] <= a[1:])


def compute_pseudoage(birth_year: int) -> int:

    dob = datetime.date(birth_year, 1, 1)
    end = datetime.date(2007, 1, 1)

    difference_in_years = relativedelta(end, dob).years
    return difference_in_years
