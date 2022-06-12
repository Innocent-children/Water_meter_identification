from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session, flash, g
from flask_mail import Message
from exts import mail, db
from models import *
import string
import random
from .forms import RegisterForm, LoginForm
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint('user', __name__, url_prefix='/user')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        form = LoginForm(request.form)
        if form.validate():
            email = form.email.data
            password = form.password.data
            user = UserModel.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                if user.auto == 1:
                    return redirect('/tmp')
                return redirect('/')
            else:
                flash('邮箱或密码错误！')
                return redirect(url_for('user.login'))
        else:
            flash('邮箱或密码错误！')
            return redirect(url_for('user.login'))


@bp.route("/logout")
def logout():
    # 清除session中所有的数据
    session.clear()
    return redirect(url_for('user.login'))


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        form = RegisterForm(request.form)
        if form.validate():
            email = form.email.data
            username = form.username.data
            password = form.password.data
            hash_password = generate_password_hash(password)
            if password == '654321':
                auto = 1
            else:
                auto = 0
            user = UserModel(email=email, username=username, password=hash_password, auto=auto)
            db.session.add(user)
            db.session.commit()
            flash('注册成功！即将跳转到登录页面')
            return redirect(url_for('user.login'))
        else:
            flash('信息有误！')
            return redirect(url_for('user.register'))


@bp.route('/captcha', methods=['POST'])
def get_captcha():
    email = request.form.get('email')
    letters = string.ascii_letters + string.digits
    captcha = "".join(random.sample(letters, 6))
    if email:
        messages = Message(
            subject='【水表系统验证码】',
            recipients=[email],
            body=f'【水表系统】\n您好，您的验证码为\n【{captcha}】\n请不要告诉任何人'
        )
        mail.send(messages)
        captcha_model = EmailCaptchaModel.query.filter_by(email=email).first()
        if captcha_model:
            captcha_model.captcha = captcha
            captcha_model.create_time = datetime.now()
            db.session.commit()
        else:
            captcha_model = EmailCaptchaModel(email=email, captcha=captcha)
            db.session.add(captcha_model)
            db.session.commit()
        return jsonify({'code': 200})
    else:
        return jsonify({'code': 400, 'messages': '请输入邮箱'})
