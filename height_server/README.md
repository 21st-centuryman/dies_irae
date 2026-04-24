<div align="center">

## Height server

#### A server that fetches height data based on Lantmäteret api

[![Python](https://img.shields.io/badge/python-3776AB.svg?style=for-the-badge&logoColor=white&logo=python)]()

</div>

## Usage

First you need to start the python server with the following command

```sh
python3 height.py
```

This starts a http server serving port 8000.

You can either write the data or fetch it directly using the following commands

```sh
curl 'http://localhost:8000/fetch?lat=56.579&lon=14.186'
curl 'http://localhost:8000/write?lat=56.579&lon=14.186&out=./terrain.bin'
```

Note: you need to add a `.env` with the following information:

* API_KEY: String
* RADIUS_KM: Float
* DEFAULT_TIMEOUT: Int
* OUTPUT_SIZE: Int
