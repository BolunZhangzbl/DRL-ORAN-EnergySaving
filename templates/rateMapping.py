
'''
From 38.214 5G NR CQI table 2
'''
CQI_to_rate = {15: 84 * 7.40,
               14: 84 * 6.91,
               13: 84*6.23 ,
               12: 84*5.55,
               11: 84*5.12, 
               10: 84 *4.52, 
               9: 84 *3.90,
               8: 84 *3.32,
               7: 84 *2.73, 
               6: 84 *2.41,
               5: 84 *1.91,
               4: 84 *1.48, 
               3: 84 *0.88, 
               2: 84 *0.38, 
               1: 84 *0.15, 
               0: 0}


'''
Quantize SNR_db value according to the link simulation under rician channel
'''
CQI_to_threshold = {0: 0, 
                    1: 0, 
                    2: 0, 
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 5,
                    7: 10, 
                    8: 13, 
                    9: 15,

                    10: 20,
                    11: 25,
                    12: 30,
                    13: 35,
                    14: 40,
                    15: 45 }


def snr_to_CQI (snr_db):
    # map snr to CQI level
    # this part is set by simulation

    if snr_db <= 0:
        CQI = 4
    elif 0 < snr_db <= 5:
        CQI = 5
    elif 5 < snr_db <= 10:
        CQI = 6
    elif 10 < snr_db <= 13:
        CQI = 7
    elif 13 < snr_db <= 15:
        CQI = 8
    elif 15 < snr_db <= 20:
        CQI = 9
    elif 20 < snr_db <= 25:
        CQI = 10

    elif 25 < snr_db <= 30:
        CQI = 11
    elif 30 < snr_db <= 35:
        CQI = 12    
    elif 35 < snr_db <= 40:
        CQI = 13 
    elif 40 < snr_db <= 45:
        CQI = 14
    else:
        CQI = 15
    return CQI


