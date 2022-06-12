# 数据库配置变量
HOSTNAME = 'localhost'
PORT = '3306'
DATABASE = 'water'
USERNAME = 'root'
PASSWORD = '12345'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(USERNAME, PASSWORD,
                                                              HOSTNAME, PORT,
                                                              DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI
JSON_AS_ASCII = False
SECRET_KEY = '123'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MAIL_SERVER = 'SMTP.qq.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_DEBUG = True
MAIL_USERNAME = 'gjc.aixuexi@foxmail.com'
MAIL_PASSWORD = 'hlfdittrhjgifdeb'
MAIL_DEFAULT_SENDER = 'gjc.aixuexi@foxmail.com'
