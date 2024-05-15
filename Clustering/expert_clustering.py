

from kmodes.kprototypes import KPrototypes
import pandas as pd
import numpy as np




class Clustering_expert():
    def __init__(self, data):
        self.data = data
        self.table = [
            (0.00000000000000, 0.45068731085587, "0%"),
            (0.45068731085587, 0.53693152638850, "1%"),
            (0.53693152638850, 0.56012914599981, "2%"),
            (0.56012914599981, 0.57594408135238, "3%"),
            (0.57594408135238, 0.59019546284842, "4%"),
            (0.59019546284842, 0.60390915137906, "5%"),
            (0.60390915137906, 0.61387036565312, "6%"),
            (0.61387036565312, 0.62571423608426, "7%"),
            (0.62571423608426, 0.63467545579118, "8%"),
            (0.63467545579118, 0.64427378967024, "9%"),
            (0.64427378967024, 0.65244952181274, "10%"),
            (0.65244952181274, 0.66184657879897, "11%"),
            (0.66184657879897, 0.67125814175790, "12%"),
            (0.67125814175790, 0.68047193193158, "13%"),
            (0.68047193193158, 0.68830555161621, "14%"),
            (0.68830555161621, 0.69666020072924, "15%"),
            (0.69666020072924, 0.70388414439727, "16%"),
            (0.70388414439727, 0.71153298410291, "17%"),
            (0.71153298410291, 0.71899454804265, "18%"),
            (0.71899454804265, 0.72842289189164, "19%"),
            (0.72842289189164, 0.73652249238699, "20%"),
            (0.73652249238699, 0.74543636742703, "21%"),
            (0.74543636742703, 0.75351564051001, "22%"),
            (0.75351564051001, 0.76215264341252, "23%"),
            (0.76215264341252, 0.77078297978634, "24%"),
            (0.77078297978634, 0.77778005227287, "25%"),
            (0.77778005227287, 0.78611868125808, "26%"),
            (0.78611868125808, 0.79452033073732, "27%"),
            (0.79452033073732, 0.80227687182006, "28%"),
            (0.80227687182006, 0.81068840950830, "29%"),
            (0.81068840950830, 0.81940279406454, "30%"),
            (0.81940279406454, 0.82863334797177, "31%"),
            (0.82863334797177, 0.83699138271781, "32%"),
            (0.83699138271781, 0.84490126552511, "33%"),
            (0.84490126552511, 0.85263689336860, "34%"),
            (0.85263689336860, 0.86220561525508, "35%"),
            (0.86220561525508, 0.87170736326378, "36%"),
            (0.87170736326378, 0.88042590645531, "37%"),
            (0.88042590645531, 0.88934192133808, "38%"),
            (0.88934192133808, 0.89803565732055, "39%"),
            (0.89803565732055, 0.90760546566724, "40%"),
            (0.90760546566724, 0.91711399269845, "41%"),
            (0.91711399269845, 0.92665541420424, "42%"),
            (0.92665541420424, 0.93482914035115, "43%"),
            (0.93482914035115, 0.94504238184167, "44%"),
            (0.94504238184167, 0.95320582391708, "45%"),
            (0.95320582391708, 0.96243427199884, "46%"),
            (0.96243427199884, 0.97141500347254, "47%"),
            (0.97141500347254, 0.97960925718515, "48%"),
            (0.97960925718515, 0.99082682863388, "49%"),
            (0.99082682863388, 1.00000000000001, "50%"),
            (1.00000000000001, 1.01065485901958, "51%"),
            (1.01065485901958, 1.02157391917041, "52%"),
            (1.02157391917041, 1.03181408023206, "53%"),
            (1.03181408023206, 1.04290709369499, "54%"),
            (1.04290709369499, 1.05300343580120, "55%"),
            (1.05300343580120, 1.06334382623829, "56%"),
            (1.06334382623829, 1.07439584185308, "57%"),
            (1.07439584185308, 1.08577986242574, "58%"),
            (1.08577986242574, 1.09544918746511, "59%"),
            (1.09544918746511, 1.10705628384487, "60%"),
            (1.10705628384487, 1.11997738822194, "61%"),
            (1.11997738822194, 1.12942030426358, "62%"),
            (1.12942030426358, 1.14185541457440, "63%"),
            (1.14185541457440, 1.15234442314801, "64%"),
            (1.15234442314801, 1.16399551477489, "65%"),
            (1.16399551477489, 1.17656325837651, "66%"),
            (1.17656325837651, 1.19018468911221, "67%"),
            (1.19018468911221, 1.20213422269647, "68%"),
            (1.20213422269647, 1.21567283358572, "69%"),
            (1.21567283358572, 1.22791608907198, "70%"),
            (1.22791608907198, 1.24141082631924, "71%"),
            (1.24141082631924, 1.25377938288031, "72%"),
            (1.25377938288031, 1.26877072165772, "73%"),
            (1.26877072165772, 1.28193361666114, "74%"),
            (1.28193361666114, 1.29853083455681, "75%"),
            (1.29853083455681, 1.31014588815420, "76%"),
            (1.31014588815420, 1.32455174430608, "77%"),
            (1.32455174430608, 1.34049975511985, "78%"),
            (1.34049975511985, 1.35731773146750, "79%"),
            (1.35731773146750, 1.37463535009464, "80%"),
            (1.37463535009464, 1.39224716050131, "81%"),
            (1.39224716050131, 1.40773678250452, "82%"),
            (1.40773678250452, 1.42548384439408, "83%"),
            (1.42548384439408, 1.44443536742566, "84%"),
            (1.44443536742566, 1.46479968014262, "85%"),
            (1.46479968014262, 1.48742806263682, "86%"),
            (1.48742806263682, 1.50968118989544, "87%"),
            (1.50968118989544, 1.53317154635861, "88%"),
            (1.53317154635861, 1.56311479808836, "89%"),
            (1.56311479808836, 1.59606064434504, "90%"),
            (1.59606064434504, 1.62623982811626, "91%"),
            (1.62623982811626, 1.66260973272395, "92%"),
            (1.66260973272395, 1.69451827350295, "93%"),
            (1.69451827350295, 1.73922305956362, "94%"),
            (1.73922305956362, 1.79071556331901, "95%"),
            (1.79071556331901, 1.85341242865515, "96%"),
            (1.85341242865515, 1.93023256569179, "97%"),
            (1.93023256569179, 2.03513854165712, "98%"),
            (2.03513854165712, 999999999.00000000000000, "99%"),
            (999999999.00000000000000, None, "100%"),
        ]
        self.table_coarse = [
            (0.00000000000000, 0.65244952181274, 1),
            (0.65244952181274, 0.73652249238699, 2),
            (0.73652249238699, 0.81940279406454, 3),
            (0.81940279406454, 0.90760546566724, 4),
            (0.90760546566724, 1.00000000000001, 5),
            (1.00000000000001, 1.10705628384487, 6),
            (1.10705628384487, 1.22791608907198, 7),
            (1.22791608907198, 1.37463535009464, 8),
            (1.37463535009464, 1.59606064434504, 9),
            (1.59606064434504, 10e13, 10),
        ]



        self.bins = [0, 0.65244952181274, 0.73652249238699, 0.81940279406454, 0.90760546566724, 1.00000000000001, 1.10705628384487, 1.22791608907198, 1.37463535009464, 1.59606064434504, 10e3]
        self.labels =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    def fit_and_encode(self):



        Xbeta = 0.0128 * (self.data['age_don'] -  40)  - (0.0194 * (self.data['age_don']- 18) *(self.data['age_don'] < 18)) + (0.0107 * (self.data['age_don'] - 50) * (self.data['age_don'] > 50))
        Xbeta -=  (0.0464 * ((self.data['height_don'] - 170)/10)) - (0.0199 * ((self.data['weight_don'] - 80)/10) * (self.data['weight_don'] < 80))
        Xbeta +=  0.1790 * (self.data['race_don']) + (0.1260 * (self.data['hypertension_don'])) + 0.1300 * (self.data['diabetes_don'])
        Xbeta +=  0.0881 * (self.data['death_cerebrovascular']) + (0.2200 * (self.data['creatinine_don'] - 1))
        Xbeta -= 0.2090 * (self.data['creatinine_don'] - 1) * (self.data['creatinine_don'] > 1.5 ) + (0.2400 * (self.data['HCV_don'])) + (0.1330 * (self.data['DCD_don']))

        KDRI_RAO = np.exp(Xbeta)
        normalizing_factor = 1.318253823684 #Should be the median KDRI from most recent cohort (taken from paper (USA, 2021))
        KDRI_median = KDRI_RAO/normalizing_factor



        labels = pd.cut(KDRI_median, bins=self.bins, labels=self.labels)
        
        return np.array(labels)

        
    





    



if __name__ == "__main__":

    organs = pd.read_csv('C:/Users/Ernesto/OneDrive - ETH Zurich/Desktop/MT/COMET/synthetic_data_generation/organs.csv')

    n_clusters = 3
    clustering = Clustering_expert(organs)
    print(clustering.fit_and_encode())
