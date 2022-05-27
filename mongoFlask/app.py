from flask import Flask, request, render_template, url_for, redirect, jsonify
from pymongo import MongoClient

app = Flask(__name__)

try:
    # mongo_uri = 'mongodb://localhost:27017/'
    client = MongoClient(
        'localhost', 27017,
        username='root',
        password='password',
        serverSelectionTimeoutMS=1000
    )
    print(client.server_info())

    db = client.testCollection
    # Can also create the database using dictionaries:
    # db = client.testCollection

except Exception as ex:
    print('Error - Cannot connect to db')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        if request.form.get("create_user"):
            return redirect(url_for('create_user'))
        elif request.form.get("list_collection_names"):
            return redirect(url_for('list_collection_names'))
        elif request.form.get("delete_by_name"):
            return redirect(url_for('delete_by_name'))
        elif request.form.get("find_all"):
            return redirect(url_for('find_all'))
        else:
            print('Need to redirect to url')


@app.route('/create_user', methods=['POST', 'GET'])
def create_user():
    if request.method == 'POST':
        item_doc = {
            'name': request.form['name'],
            'password': request.form['password']
        }
        db.testCollection.insert_one(item_doc)
        return redirect(request.url)  # reminder: this returns to the same page after submission

    elif request.method == 'GET':
        return render_template('create_user.html')


@app.route('/list_collection_names', methods=['GET'])
def list_collection_names():
    return jsonify({'output': db.list_collection_names()})


@app.route('/find_all', methods=['GET'])
def find_all():
    docs = []
    for doc in db.testCollection.find():
        docs.append(doc)
    return render_template('find_all.html', docs=docs)


@app.route('/delete_by_name', methods=['GET', 'POST'])
def delete_by_name():
    if request.method == 'GET':
        return render_template('delete_by_name.html')

    elif request.method == 'POST':
        db.testCollection.delete_one({'name': request.form['input_name']})
        print(f"Successfully deleted {request.form['input_name']}")
        return redirect(request.url)


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
