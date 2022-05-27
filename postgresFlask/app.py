import sys

from flask import Flask, render_template, url_for, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import psycopg2


def create_database():
    conn = psycopg2.connect(database="postgres", user='postgres', password='mysecretpassword', host='localhost',
                            port='5432')
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Preparing query to create a database
    sql = '''CREATE database users_database'''

    # Creating a database
    cursor.execute(sql)
    print("Database created successfully...")

    # Closing the connection
    conn.close()


try:
    create_database()
except psycopg2.OperationalError:
    print(f'Cannot connect to the database (start the docker)')
    sys.exit(1)
except psycopg2.errors.DuplicateDatabase:
    print(f'Database already exists...')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:mysecretpassword@localhost:5432/users_database"
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = 'users_table'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    email = db.Column(db.String())
    password = db.Column(db.String())

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

    def __repr__(self):
        return f'<id {self.id} | name {self.name} | email {self.email} | password {self.password}>'


# create the users table inside of the users database
db.create_all()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User(name, email, password)
        db.session.add(user)
        db.session.commit()

        return render_template('index.html')


@app.route('/find_all', methods=['GET'])
def find_all():
    users = User.query.all()
    tempDict = {}
    for idx, user in enumerate(users):
        tempDict[idx] = str(user)
    return jsonify(tempDict)


@app.route('/find_one', methods=['GET', 'POST'])
def find_one():
    # user = User.query.filter_by(name=)
    if request.method == 'GET':
        return render_template('find_one.html')
    elif request.method == 'POST':
        searchName = request.form.get('name')
        users = User.query.filter_by(name=searchName).all()
        tempDict = {}
        for idx, user in enumerate(users):
            tempDict[idx] = str(user)
        return jsonify(tempDict)


@app.route('/users_count', methods=['GET'])
def users_count():
    count = User.query.count()
    return jsonify({'users_count': count})


if __name__ == '__main__':
    app.run(debug=True)
