# DIC Exercise 3 - Group 40

- Oto Alves - 12045798
- Artjola Ganellari - 12046001
- Lukas Mölschl 01418388

## Docker

Build docker image from Dockerfile
`$ docker build -t dic-assignment-3-group-40 .`

Run docker container locally
`$ docker run -v $PWD/images:/work -p 5040:5000 dic-assignment-3-group-40:latest`

## Run inference

On a single file
`$ curl http://localhost:5040/api/detect -d "input=/work/images/filename.jpg"`

Run on a single file and save the bounding boxes to the `out/` directory on the server
`$ curl http://localhost:5040/api/detect -d "input=/work/images/filename.jpg&output=1"`

Run on a directory with images (optionally with the `output=1` flag)
`$ curl http://localhost:5040/api/detect -d "input=/work/images/"`

## Upload Server

Start the [upload server](https://hub.docker.com/r/mayth/simple-upload-server/)
`$ docker run -p 5040:25478 -v $HOME/tmp:/var/root mayth/simple-upload-server -token mytoken -upload_limit 7000000000 /var/root`
`-upload_limit` in bytes that the server maximally accepts before closing the connection
`-token` a _secure_ token that is required when uploading a file

Upload a single file
`$ curl -Ffile=@filename.jpg 'http://s25.lbd.hpc.tuwien.ac.at:5040/upload?token=mytoken'`
