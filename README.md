# Investigation on production deployment of ML models

In case there are older codes that use `tensorflow 1.x`, it will require python version `< 3.8`. Ubuntu 20.04 comes with `Python 3.8` as default and its quite cumbersome to install different versions of python.

## Managing multiple Python versions

From ubuntu 20.04, it does not allow installation of other python version than the default (3.8).

To install a differen version (for example 3.6), Deadsnakes PPA is useful.
```
$ sudo add-apt-repository ppa:deadsnakes/ppa

# Run update
$ sudo apt update

# Install your required version of python
$ sudo apt install python3.x
```

To install virtual environment, python3.x-venv is required.
```
$ sudo apt install python3.x-apt

# Install virtual environment
$ python3.x -m venv venv

# Activate the environment
$ source venv/bin/activate
```

## Saving trained tensorflow model

Right way to save the tensorflow model is to save the weights generated.

If `model` is the name of variable that holds tensorflow trained model, it is saved to file `model.hd5` as:
```
model.save("model.h5")
```

Ship the model to another platform/machine and it can be loaded into system as following:
```
remote_load_model = tf.keras.models.load_model("model.hd5")
```

## Issues encountered when serving the prediction from saved model

FastApi server was not serving numpy float. ASGI server was breaking. The issue got solved by converting the numpy float to string. It can also be resolved by converting numpy float to python-float .i.e.
```
>>> x = numpy.array([1.1, 2.2, 3.3])
>>> type(x)
<class 'numpy.float64'>
>>> type(x[0])
<class 'numpy.float64'>
>>> y = float(x[0])
>>> type(y)
<class 'float'>
```
