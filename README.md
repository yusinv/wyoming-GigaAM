# Wyoming gigaAM

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [gigaAM](https://github.com/salute-developers/GigaAM/) speech to text system.

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/yusinv/wyoming-GigaAM.git
cd wyoming-GigaAM
script/setup
```

Run a server anyone can connect to:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data
```

## Docker Image

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data yusinv/wyoming-gigaam
```

