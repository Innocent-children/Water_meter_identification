from turtle import back

from sqlalchemy.orm import backref

from exts import db
from datetime import datetime


class EmailCaptchaModel(db.Model):
    __tablename__ = 'email_captcha'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    captcha = db.Column(db.String(10), nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)


class UserModel(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    join_time = db.Column(db.DateTime, default=datetime.now)
    auto = db.Column(db.Integer)


class PictureModel(db.Model):
    __tablename__ = 'picture'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pic = db.Column(db.BLOB, nullable=True)
    pic_name = db.Column(db.String(100), nullable=False)
    result = db.Column(db.Integer, nullable=False, default=000000)
    # uploader_id = db.Column(db.Integer, db.ForeignKey('user.id'))
