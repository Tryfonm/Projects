start a mysql server with docker:
```
docker-compose -f  docker-compose.yml up -d
```
enter the continaier with:
```
docker exec -it /bin/bash
```
inside the container enter the database with:
```
 mysql -u mysql_user -p mysql_database
```
and use password (as defined in the .yml file):
```
mysecretpasssword
```
to gracefully detach from the container if started without '-d' flag :
```
Ctrl+Q
```

endpoints:
```
http://localhost:5000/
http://localhost:5000/find_all
http://localhost:5000/find_one
http://localhost:5000/users_count