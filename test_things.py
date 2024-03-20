import pickle
import copy


global long_name_var
long_name_var = None


def func1():
    with open('/tmp/data/training_result.pkl', 'rb') as file:
        local_model_params, loss, accuracy = pickle.load(file)

    return local_model_params

long_name_var = func1()
print(long_name_var)
