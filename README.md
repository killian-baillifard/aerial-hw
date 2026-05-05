# Aerial robotics hardware project

## Getting started

Download this repository via HTTP
```shell
git clone --recurse-submodules -j8 https://github.com/killian-baillifard/aerial-hw.git
```

or SSH (will require a public / private key pair logged in your account)
```shell
git clone --recurse-submodules -j8 git@github.com:killian-baillifard/aerial-hw.git
```

Enter the repository
```shell
cd aerial-hw
```

## Virtual environnement (optionnal)

Create a virtual environnement
```shell
python -m venv .env
```

Activate it on Windows
```shell
.\.env\Scripts\activate
```

or Mac / Linux
```shell
source ./.env/bin/activate
```

## Launch the application

Install the crazyflie library
```
cd crazyflie-lib-python
pip install -e .
```

Install all requirements using
```shell
cd ../app
pip install -r requirements.txt
```

Start application as a python module
```shell
cd ..
python -m app
```
