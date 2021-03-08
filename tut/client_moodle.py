import json
import requests
import numpy as np

######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.156'
PORT = 3000
MAX_DEG = 11

#### functions that you can call
def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert -10<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

#### utility functions
def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


if __name__ == "__main__":
    """
    Replace "test" with your secret ID and just run this file 
    to verify that the server is working for your ID.
    """
    # vec = [
    #                 0.0,
    #                 0.0,
    #                 -0.059653459864157016,
    #                 0.04927146068222989,
    #                 0.0,
    #                 0.0,
    #                 0.0,
    #                 8.810393206711616e-10,
    #                 0.0,
    #                 2.4620984973162587e-13,
    #                 0.0
    #             ]
    # err = get_errors('qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow', vec)
    # print(err[0], err[1], err[0]+ err[1])
    # assert len(err) == 2
    vec = [
        0.0,
        -1.4863007867815543e-12,
        -2.439965013525661e-13,
        5.4377571962439426e-11,
        -1.6830096260138962e-10,
        -1.9159070877831255e-15,
        3.6069588865212196e-16,
        2.3065419829270534e-05,
        -2.04721003e-06,
        -1.5648322605669267e-08,
        9.855799881333486e-10
    ]
    vec = [
        0.0,
        -1.4452806349176664e-12,
        -2.28980078e-13,
        4.733053052274488e-11,
        -1.7990579166858492e-10,
        -1.8366977e-15,
        8.5294406e-16,
        2.29423303e-05,
        -2.04721003e-06,
        -1.59792834e-08,
        9.98214034e-10
    ]

    vec = [
        0.0,
        -1.478326011643276e-12,
        -2.228133085568505e-13,
        5.306217689228875e-11,
        -1.9311708259317407e-10,
        -1.6555686257132759e-15,
        7.164179564288258e-16,
        2.3092938538014106e-05,
        -1.8889828756880271e-06,
        -1.5282730159004693e-08,
        9.027492857847415e-10
    ]
    vec = [
        0.0,
        -1.4002108983525113e-12,
        -2.2828952824723084e-13,
        4.727434750341003e-11,
        -1.8831312078855928e-10,
        -1.5364734734849751e-15,
        3.775570321573971e-16,
        2.2714020299470102e-05,
        -1.8888426805555975e-06,
        -1.5282730439247565e-08,
        9.027492857955074e-10
    ]

    vec = [
        0.0,
        -1.3971220652144584e-12,
        -2.2547458377265118e-13,
        4.474826845262611e-11,
        -1.6831494108309235e-10,
        -1.3844164647257515e-15,
        2.5972683717004705e-16,
        2.267159826896988e-05,
        -1.8967996700219494e-06,
        -1.528273047066431e-08,
        9.027492857967142e-10
    ]

    #vec = [
    #    0.0,
    #    -1.4727752866632174e-12,
    #    -2.1984711428221744e-13,
    #    4.6201075299999997e-11,
    #    -1.7416926105420417e-10,
    #    -2.2638608445245176e-15,
    #    7.682152646886355e-16,
    #    2.3071895681795815e-05,
    #    -2.04721003e-06,
    #    -1.5979283399999998e-08,
    #    9.982140339999997e-10
    #]
    # vec = [
    #     0.0,
    #     -1.4057516938785075e-12,
    #     -2.1986579292007392e-13,
    #     4.6206363511348445e-11,
    #     -1.7420646893210177e-10,
    #     -2.2556308378507234e-15,
    #     7.611941253625271e-16,
    #     2.3069399389757186e-05,
    #     -2.04721003e-06,
    #     -1.598042274945641e-08,
    #     9.982140339999997e-10
    # ]

    print(submit('qF9wD9XQoX5Pzca7gYOPAQxP6eUozTMGIelPiQVnV8XiqKeVow', vec))

    # submit_status = submit('dnLVLTHPAUOT2R1Ruj1sQvXxWBZZchp8u4WkyZGzaeTQCpyFXC', list(-np.arange(0,1.1,0.1)))
    # assert "submitted" in submit_status

