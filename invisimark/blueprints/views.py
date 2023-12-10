from flask import render_template, request, redirect, url_for, flash, send_file, send_from_directory
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from invisimark.services.dct_image import DCTImage
from invisimark.services.dct_text import DCTText
from invisimark.services.dwt_image import DWTImage
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import uuid

users = [
    {"id": "91d533a1-2c26-433c-b366-09a62232a822", "name": "Cid Kagenou", "email": "user1@example.com", "password": "password1"},
    {"id": "79b1e780-f193-4996-aa62-3717eaf9a354", "name": "Arthur Leywin", "email": "user2@example.com", "password": "password2"},
]

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(os.path.dirname(
    current_directory))
image_directory = 'images/insertion'

UPLOAD_FOLDER = os.path.join(project_directory, image_directory)


def init_app(app):
    app.secret_key = 'super secret key'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    login_manager = LoginManager()
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        for user_data in users:
            if user_data['id'] == user_id:
                u = User()
                u.id = user_data['id']
                u.name = user_data['name']
                return u
        return None

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']

            for user in users:
                if user['email'] == email and user['password'] == password:
                    u = User()
                    u.id = user['id']
                    login_user(u)
                    flash('Login bem-sucedido!', 'success')
                    return redirect(url_for('dashboard'))

            flash('Credenciais inválidas')

        return render_template('auth/login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']

            for user in users:
                if user['email'] == email and user['password'] == password:
                    return redirect(url_for('/'))

            return "Credenciais inválidas"

        return render_template('auth/register.html')

    @app.route('/dashboard')
    @login_required
    def dashboard():
        user_images = current_user.images
        return render_template('dashboard/index.html', username=current_user.name, user_images=user_images)

    @app.route('/dashboard/insertion', methods=['GET', 'POST'])
    @login_required
    def insertion():
        if request.method == 'POST':
            if 'image' not in request.files:
                flash('Nenhum arquivo enviado')
                return redirect(request.url)

            file = request.files['image']
            insertion_type = request.form['insertion_type']
            watermark_file = request.files.get('watermark_file')
            watermark_text = request.form.get('watermark_text')

            if file.filename == '':
                flash('Nenhum arquivo selecionado')
                return redirect(request.url)

            watermark_file = cv2.imdecode(np.fromstring(
                watermark_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            if file and allowed_file(file.filename):
                user_filename = f"{current_user.id}-{str(uuid.uuid4())}.png"

                file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
                file.save(file_path)

                original_image = cv2.imread(file_path)

                marked_image = perform_insertion(
                    original_image, insertion_type, 0.1, watermark_file, watermark_text)

                if marked_image is not None:
                    cv2.imwrite(file_path, marked_image)

                    current_user.images.append(file_path)

                    return send_file(file_path, as_attachment=True)

            else:
                flash('Extensão de arquivo inválida')

            return redirect(url_for('dashboard'))

        return render_template('dashboard/insertion.html')

    @app.route('/dashboard/myprofile')
    @login_required
    def myprofile():
        return render_template('dashboard/myprofile.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_insertion(original_image, insertion_type, alpha=0.1, watermark=None, text=None):
    if insertion_type == 'image_dct':
        marked_image = DCTImage.rgb_insert_dct_blocks(
            original_image, watermark, alpha)
    elif insertion_type == 'image_dwt':
        marked_image = DWTImage.embed_watermark_HH_blocks(
            original_image, watermark, alpha)
    elif insertion_type == 'text_dct':
        marked_image = DCTText.rgb_insert_texto(original_image, text, alpha)
    else:
        flash('Tipo de inserção não suportado')
        return None

    return marked_image


class User:
    def __init__(self, user_id=None, email=None, password=None):
        self.id = user_id
        self.email = email
        self.password = password
        self.image = []

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)
