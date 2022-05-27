run postgres docker with:
```
docker run --name some-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d postgres
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