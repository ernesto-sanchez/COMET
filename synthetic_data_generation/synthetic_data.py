import numpy as np
import random
import pandas as pd


def generate_patient(pat_id: int) -> array:

    """
    Generate synthetic data for a transplant patient and donor. See comments for details on the data generation process. !!!!! Every feature is generated independently !!!!!!!!
    """
    age = random.randint(20, 80)  # Age between 20 and 80
    sex = random.choice(["male", "female"])  # Sex, 0 for Male and 1 for Female
    blood_type = random.choice(["A", "B", "AB", "0"])  # Blood Type, 1 for 'A', 2 for 'B', 3 for 'AB', 4 for 'O'
    rh = random.choice(["+", "-"])  # Rh, 0 for Negative and 1 for Positive
    weight  =  max(40, min(120, round(random.gauss(75,10),2))) # Weight normally distributed with a mean on 75 between 50 and 100 kg with 2 decimal places
    hla_a = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'
    hla_b = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'
    hla_c = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'
    

    return [pat_id, age, sex, blood_type, rh, weight, hla_a, hla_b, hla_c]




def generate_organ(org_id: int) -> array:
    """
    Generate synthetic data for a transplant organ. See comments for details on the data generation process. 

    """
    cold_ischemia_time = max(0, min(round(random.gauss(7, 2),2),24))  # Cold Ischemia Time between 0 and 24 hours, gaussian distribution with a mean of 7 hours and a standard deviation of 2 hours
    dsa = random.choice([0, 1])  # DSA, 0 for Negative and 1 for Positive
    blood_type_don = random.choice(["A", "B", "AB", "0"])  # Blood Type, 1 for 'A', 2 for 'B', 3 for 'AB', 4 for 'O'
    rh_don = random.choice(["+", "-"])  # Rh, 0 for Negative and 1 for Positive
    age_don = random.randint(20, 80)  # Age between 20 and 80
    sex_don = random.choice(["male", "female"])
    hla_a_don = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'
    hla_b_don = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'
    hla_c_don = random.choice(range(1, 5))  # HLA, 1 for 'A', 2 for 'B', 3 for 'C', 4 for 'D'


    return [org_id, cold_ischemia_time, dsa, blood_type_don, rh_don, age_don, sex_don, hla_a_don, hla_b_don, hla_c_don]



def generate_outcomes(id_match:int, features_pat: array, features_org: array) -> array:
    """
    Generate (factual and counterfactual) outcomes for a transplant patient. See comments for details on the data generation process. 
    """
    

    # TODO: implement logic to generate the outcomes

    
    return [id_match, e_gfr_1, e_gfr_2, e_gfr_3, e_grf_4, rej_1, rej_2, rej_3, rej_4]


def main():
    n = 100  # number of patients
    m = 100  # number of organs

    patients = [generate_patient(i) for i in range(n)]
    organs = [generate_organ(i) for i in range(m)]


    # Create a dataframe with the synthetic data
    df_patients = pd.DataFrame(patients, columns=["pat_id", "age", "sex", "blood_type", "rh", "weight", "hla_a", "hla_b", "hla_c"])
    df_organs = pd.DataFrame(organs, columns=["org_id", "cold_ischemia_time", "dsa", "blood_type_don", "rh_don", "age_don", "hla_a_don", "hla_b_don", "hla_c_don"])


    # Save the data as a csv file
    df_patients.to_csv('patients.csv', index=False)
    df_organs.to_csv('organs.csv', index=False)

    #print(df)



if __name__ == '__main__':
    main()
