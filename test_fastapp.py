"""
test file to test fast api GET and POST requests
    steps:
    1. run function run_tests to run customized tests(optional to retrain)
    2. Or run pytest

    author: Dr. Sage Chhetri
"""
import requests
import os
import pandas as pd
import constants
import test_data

RETRAIN = False
print("Running Tests - please wait ... ")


def test_get_home_data1():
    """
    postive test case to test first data set for welcome message
    """

    if constants.ENV == 'DEV':
        get_home_url = constants.get_home_url_dev

    elif constants.ENV == 'LIVE':
        get_home_url = constants.get_home_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    home_message = test_data.test_get_data1['get_home_message']
    msg = "Testing home url and message"
    msg = f"{msg}\nusing test data = {home_message}"
    response = requests.get(get_home_url)

    if not response.status_code == 200:
        msg = f"{msg}\n\tFAILED :Test test_get_home failed."
        msg = f"{msg}\n\tNo access {get_home_url},{response.status_code}"

    else:
        msg = f"{msg}\n\tPASSED :Test passed for status code check."
        msg = f"{msg}\n\tGot {response.status_code} at {get_home_url}"

        try:
            assert response.json()['message'] == home_message
        except AssertionError:
            msg = f"{msg}\n\tFAILED : Test failed. Wrong message"
            msg = f"{msg}\n\tExpected '{home_message}'"
            msg = f"{msg} but got '{response.json()['message']}'"

        except Exception as excep:
            msg = f"{msg}\n*** Exception occurred . {str(excep)}"

    return msg, response.json()


def test_get_home_data2():
    """
    megative test case to test first data set for wrong welcome message
    """

    if constants.ENV == 'DEV':
        get_home_url = constants.get_home_url_dev

    elif constants.ENV == 'LIVE':
        get_home_url = constants.get_home_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    home_message = test_data.test_get_data2['get_home_message']
    msg = "Testing home url and message"
    msg = f"{msg}\nusing test data = {home_message}"
    response = requests.get(get_home_url)

    if not response.status_code == 200:
        msg = f"{msg}\n\tFAILED :Test failed."
        msg = f"{msg}\n\tNo access {get_home_url}"
        msg = f"{msg},{response.status_code}"

    else:
        msg = f"{msg}\n\tPASSED :Test passed for status code."
        msg = f"{msg}\n\tGot {response.status_code} at {get_home_url}"

        try:
            assert response.json()['message'] == home_message
        except AssertionError:
            msg = f"{msg}\n\tFAILED : Test failed. Wrong message"
            msg = f"{msg}\n\tExpected '{home_message}'"
            msg = f"{msg}\n\tgot '{response.json()['message']}'"

        except Exception as excep:
            msg = f"{msg}\n*** Exception occurred.{str(excep)}"

    return msg, response.json()


def test_post_modelinfer_data1():
    """
    postive test case- correct input file ,train and save results
    input:
        POST Method from past api
    outputs:
        saves all models in model folder
        saves all results (scores/reports) in results folder
    """
    if constants.ENV == 'DEV':
        post_modelinfer_url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        post_modelinfer_url = constants.post_modelinfer_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    data = test_data.test_post_data1
    msg = "Testing train and results"
    msg = f"{msg}\nusing test data = {data}"
    response = requests.post(post_modelinfer_url, json=data)
    if not response.status_code == 200:
        msg = f"{msg}\n\tFAILED :Test test_post_modelinfer failed."
        msg = f"{msg}\n\tNo access {post_modelinfer_url}"
        msg = f"{msg}, {response.status_code}"

    else:
        msg = f"{msg}\n\tPASSED :Test passed for status code check."
        msg = f"{msg}\n\tGot {response.status_code} at {post_modelinfer_url}"

        models = [
            "knn.pkl",
            "dt.pkl",
            "rf.pkl",
            "adb.pkl",
            "svm.pkl",
            "gdboost.pkl",
            "xgboost.pkl"]

        msg = f"{msg}\nTesting {models} saved or not:"

        for model in models:
            model_pth = f"models/{model}"

            try:
                assert os.path.exists(model_pth)
                msg = f"{msg}\n\tPASSED : {model_pth} saved successfully"

            except AssertionError:
                msg = f"{msg}\n\t*** FAILED : {model_pth} not saved"

            except Exception as excep:
                msg = f"{msg}\n*** Exception occurred.{str(excep)}"

        results = [
            "classification_scores.csv",
            "best_classification_scores.csv",
            "roc_auc_scores.csv",
            "best_roc_auc_scores.csv",
            "cross_val_scores_mean.csv",
            "best_cross_val_scores_mean.csv",
            "rf_report.csv"]
        msg = f"{msg}\nTesting {results} saved or not:"
        for resul in results:
            resul_pth = f"results/{resul}"
            try:
                assert os.path.exists(resul_pth)
                msg = f"{msg}\n\tPASSED : {resul} saved successfully"

            except AssertionError:
                msg = f"{msg}\n\t*** FAILED : {resul} not saved"

            except Exception as excep:
                msg = f"{msg}\n*** Exception occurred.{str(excep)}"

    return msg, response.json()


def test_post_modelinfer_data2():
    """
    postive test case to get correct  file, params , train and save results
    input:
        POST Method from past api
    outputs:
        saves all models in model folder
        saves all results (scores/reports) in results folder
    """

    if constants.ENV == 'DEV':
        post_modelinfer_url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        post_modelinfer_url = constants.post_modelinfer_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    data = test_data.test_post_data2
    msg = "Testing train and results"
    msg = f"{msg}\nusing test data = {data}"
    response = requests.post(post_modelinfer_url, json=data)
    if not response.status_code == 200:
        msg = f"{msg}\n\tFAILED :Test test_post_modelinfer failed. "
        msg = f"{msg}\n\tNo access {post_modelinfer_url},"
        msg = f"{msg},{response.status_code}"

    else:
        msg = f"{msg}\n\tPASSED :Test passed for status code check. "
        msg = f"{msg}\n\tGot {response.status_code} at {post_modelinfer_url}"

        models = [
            "knn.pkl",
            "dt.pkl",
            "rf.pkl",
            "adb.pkl",
            "svm.pkl",
            "gdboost.pkl",
            "xgboost.pkl"]

        msg = f"{msg}\nTesting {models} saved or not:"

        for model in models:
            model_pth = f"models/{model}"

            try:
                assert os.path.exists(model_pth)
                msg = f"{msg}\n\tPASSED : {model_pth} saved successfully"

            except AssertionError:
                msg = f"{msg}\n\t*** FAILED : {model_pth} not saved"

            except Exception as excep:
                msg = f"{msg}\n*** Exception occurred. {str(excep)}"

        results = [
            "classification_scores.csv",
            "best_classification_scores.csv",
            "roc_auc_scores.csv",
            "best_roc_auc_scores.csv",
            "cross_val_scores_mean.csv",
            "best_cross_val_scores_mean.csv",
            "rf_report.csv"]
        msg = f"{msg}\nTesting {results} saved or not:"
        for resul in results:
            resul_pth = f"results/{resul}"
            try:
                assert os.path.exists(resul_pth)
                msg = f"{msg}\n\tPASSED : {resul} saved successfully"

            except AssertionError:
                msg = f"{msg}\n\t*** FAILED : {resul} not saved"

            except Exception as excep:
                msg = f"{msg}\n*** Exception occurred.{str(excep)}"

    return msg, response.json()


def test_post_modelinfer_inputdata_data4():
    """
    test case to validate input file if file exists,
        and validate passed params and saved files
    input:
        POST Method from past api
    outputs:
        saves all models in model folder
        saves all results (scores/reports) in results folder
    """
    data = test_data.test_post_data4
    msg = "test data ={data}"
    if constants.ENV == 'DEV':
        post_modelinfer_url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        post_modelinfer_url = constants.post_modelinfer_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    msg = "{msg}\nTesting input file"
    file_pth = f"data/{data['input_csv_filename']}"
    try:
        assert os.path.exists(file_pth)
        success = True
        msg = f"{msg}\n\tPASSED: file found, {file_pth}"
    except AssertionError:
        success = False
        msg = f"{msg}\n\tFAILED: No file found, {file_pth}"

    if success:
        FALED = False
        msg = f"{msg}\nTesting rows,columns {data['input_csv_filename']}"
        df = pd.read_csv(file_pth)
        df.columns = [x.replace(" ", "") for x in df.columns]
        for col in df.columns.tolist():
            df[col] = df[col].astype(str).str.replace(" ", "")
        if df.shape[0] == 0:
            msg = f"{msg}\n\tFAILED: zero rows "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: rows {df.shape[0]} found"

        if df.shape[1] == 0:
            msg = f"{msg}\n\tFAILED: zero columns "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: columns {df.shape[1]} found"

        msg = f"{msg}\nTesting columns if {data['input_csv_filename']}"
        for col in constants.cat_features:
            if col not in df.columns:
                msg = f"{msg}\n\tFAILED: {col} missing"
                FALED = True
            else:
                msg = f"{msg}\n\tPASSED: {col} found"

        msg = f"{msg}\nTesting input parameters "
        if data['test_size'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong test_size {data['test_size']}"
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: test_size is good"

        if data['random_state'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong random_state {data['random_state']}"
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: random_state is good"

        if data['n_splits'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong n_splits {data['n_splits']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: n_splits is good"

        if data['shuffle'] not in [True, False]:
            msg = f"{msg}\n\tFAILED: wrong shuffle{data['shuffle']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: shuffle is good"

        if data['cv'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong cv {data['cv']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: cv is good"

        if FALED:
            results = [
                "classification_scores.csv",
                "best_classification_scores.csv",
                "roc_auc_scores.csv",
                "best_roc_auc_scores.csv",
                "cross_val_scores_mean.csv",
                "best_cross_val_scores_mean.csv",
                "rf_report.csv"]
            for resul in results:
                resul_pth = f"results/{resul}"
                if os.path.exists(resul_pth):
                    os.remove(resul_pth)

            response = requests.post(post_modelinfer_url, json=data)
            msg = f"{msg}\n\t{response} from post {post_modelinfer_url}"
            msg = f"{msg}\n\n\tAs above params/file filed- results not saved:"
            for resul in results:
                resul_pth = f"results/{resul}"
                try:
                    assert os.path.exists(resul_pth)
                    msg = f"{msg}\n\tFAILED : {resul} saved successfully"

                except AssertionError:
                    msg = f"{msg}\n\t*** PASSED : {resul} not saved"

                except Exception as excep:
                    msg = f"{msg}\n*** Exception occurred. {str(excep)}"

    return msg


def test_post_modelinfer_inputdata_data3():
    data = test_data.test_post_data3
    msg = "test data ={data}"
    if constants.ENV == 'DEV':
        post_modelinfer_url = constants.post_modelinfer_url_dev

    elif constants.ENV == 'LIVE':
        post_modelinfer_url = constants.post_modelinfer_url_live

    else:
        print("Env flag is set wrong in constants.py file")

    msg = "{msg}\nTesting input file"
    file_pth = f"data/{data['input_csv_filename']}"
    try:
        assert os.path.exists(file_pth)
        success = True
        msg = f"{msg}\n\tPASSED: file found, {file_pth}"
    except AssertionError:
        success = False
        msg = f"{msg}\n\tFAILED: No file found, {file_pth}"

    if success:
        FALED = False
        msg = f"{msg}\nTesting rows,columns of {data['input_csv_filename']}"
        df = pd.read_csv(file_pth)
        df.columns = [x.replace(" ", "") for x in df.columns]
        for col in df.columns.tolist():
            df[col] = df[col].astype(str).str.replace(" ", "")
        if df.shape[0] == 0:
            msg = f"{msg}\n\tFAILED: zero rows "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: rows {df.shape[0]} found"

        if df.shape[1] == 0:
            msg = f"{msg}\n\tFAILED: zero columns "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: columns {df.shape[1]} found"

        msg = f"{msg}\nTesting columns if {data['input_csv_filename']}"
        for col in constants.cat_features:
            if col not in df.columns:
                msg = f"{msg}\n\tFAILED: {col} missing"
                FALED = True
            else:
                msg = f"{msg}\n\tPASSED: {col} found"

        msg = f"{msg}\nTesting input parameters "
        if data['test_size'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong test_size can {data['test_size']}"
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: test_size is good"

        if data['random_state'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong random_state {data['random_state']}"
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: random_state is good"

        if data['n_splits'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong n_splits {data['n_splits']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: n_splits is good"

        if data['shuffle'] not in [True, False]:
            msg = f"{msg}\n\tFAILED: wrong shuffle {data['shuffle']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: shuffle is good"

        if data['cv'] in [0, None]:
            msg = f"{msg}\n\tFAILED: wrong cv {data['cv']} "
            FALED = True
        else:
            msg = f"{msg}\n\tPASSED: cv is good"

        if FALED:
            results = [
                "classification_scores.csv",
                "best_classification_scores.csv",
                "roc_auc_scores.csv",
                "best_roc_auc_scores.csv",
                "cross_val_scores_mean.csv",
                "best_cross_val_scores_mean.csv",
                "rf_report.csv"]
            for resul in results:
                resul_pth = f"results/{resul}"
                if os.path.exists(resul_pth):
                    os.remove(resul_pth)

            response = requests.post(post_modelinfer_url, json=data)
            msg = f"{msg}\n\t{response} from post {post_modelinfer_url}"

            msg = f"{msg}\n\n\tabove params/file failed- results not saved:"
            for resul in results:
                resul_pth = f"results/{resul}"
                try:
                    assert os.path.exists(resul_pth)
                    msg = f"{msg}\n\tFAILED : {resul} saved successfully"

                except AssertionError:
                    msg = f"{msg}\n\t*** PASSED : {resul} not saved"

                except Exception as excep:
                    msg = f"{msg}\n*** Exception occurred.  {str(excep)}"

    return msg


def run_tests(RETRAIN=False):
    """
    Runs all above test cases - core run file
    input:
        RETRAIN flag False/True
    outputs:
        returns test result message
    """

    message = "\n\n1. POS -Testing 'GET' home url, right message"
    msg, data = test_get_home_data1()
    message = f"{message}\n{msg}"
    message = f"{message}\n\n2. NEG -Testing 'GET' home, wrong message"
    msg, data = test_get_home_data2()
    message = f"{message}\n{msg}"
    if RETRAIN:
        message = f"{message}\n3. POS -Testing 'POST' Train, save results"
        msg, data = test_post_modelinfer_data1()
        message = f"{message}\n{msg}"
        message = f"{message}\n4. POS -Testing 'POST' Train, save results"
        msg, data = test_post_modelinfer_data2()
        message = f"{message}\n{msg}"
    message = f"{message}\n\n5. NEG -Testing 'POST', wrong input file"
    msg = test_post_modelinfer_inputdata_data4()
    message = f"{message}\n{msg}"
    message = f"{message}\n\n6. NEG -Testing 'POST', wrong train/test params"
    msg = test_post_modelinfer_inputdata_data3()
    message = f"{message}\n{msg}"
    return message
# MESSAGE= run_tests()
# print(MESSAGE)
