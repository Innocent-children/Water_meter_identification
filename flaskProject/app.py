from flask import Flask, jsonify, url_for, request, redirect, render_template, Response, session, g
from flask_sqlalchemy import SQLAlchemy
import config
from exts import db, mail
from blueprints import qa_bp
from blueprints import user_bp
from flask_migrate import Migrate

from models import UserModel

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)
mail.init_app(app)
migrate = Migrate(app, db)
app.register_blueprint(qa_bp)
app.register_blueprint(user_bp)


# @app.route('/')
# def index():  # put application's code here
#     return 'Hello World!'


@app.before_request
def before_request():
    user_id = session.get('user_id')
    if user_id:
        try:
            user = UserModel.query.get(user_id)
            g.user = user
        except:
            g.user = None


@app.context_processor
def context_processor():
    if hasattr(g, 'user'):
        return {'user': g.user}
    else:
        return {}


if __name__ == '__main__':
    app.run()
