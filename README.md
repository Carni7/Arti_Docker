Initially it is necessary to insert the images from the folders in the backend container into the database. For that run:
```
docker exec -it arti-flask-backend-1 /bin/bash
python insertIntoDb_LostPaintings_training.py
python insertIntoDb_LostPaintings.py
python insertIntoDb.py
```

Then run the vector Creation for all artists as well as for the lost paintings via the Postman collection.
```
http://localhost:5000/createVectorsForArtist (run for each artist)
http://localhost:5000/createVectorsLostPaintings (needs to be run only once)
```
To check if endpoints in the backend container are reachable add curl and use a Postman-converted curl command to test.
To install curl in the container
```
docker exec -it arti-flask-backend-1 /bin/bash
apt-get update
apt-get install -y curl
```


Info:
Run either of those to open bash or shell of the container
docker exec -it containerId /bin/bash
docker exec -it containerId /bin/sh
