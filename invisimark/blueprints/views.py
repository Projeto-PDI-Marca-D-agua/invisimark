from flask import render_template, request, redirect, url_for, flash, send_file, send_from_directory
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from invisimark.services.dct_image import DCTImage
from invisimark.services.dct_text import DCTText
from invisimark.services.dwt_image import DWTImage
from invisimark.services.user_service import UserService
import os
import cv2
import numpy as np
import uuid

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
USERS_IMAGES = os.path.join(REPO_DIR, 'images')
MARKED_IMAGES_PATH = os.path.join(USERS_IMAGES, 'marked_images')
WATERMARKS_PATH = os.path.join(USERS_IMAGES, 'watermarks')


def init_app(app):
    app.secret_key = 'super secret key'
    app.config['MARKED_IMAGES_PATH'] = MARKED_IMAGES_PATH
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    login_manager = LoginManager()
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return UserService.load_user(user_id)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']

            user = UserService.authenticate_user(email, password)

            if user:
                login_user(user)
                flash('Login bem-sucedido!', 'success')
                return redirect(url_for('dashboard'))

            flash('Credenciais inválidas')

        return render_template('auth/login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            email = request.form['email']
            name = request.form['name']  # Added line to get the 'name' field
            password = request.form['password']

            result = UserService.register(email, name, password)

            if result == "success":
                flash('Registration successful!', 'success')
                return redirect(url_for('login'))
            else:
                flash(result)

        return render_template('auth/register.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Logout bem-sucedido!', 'success')
        return redirect(url_for('index'))

    @app.route('/dashboard')
    @login_required
    def dashboard():
        user_images = current_user.images
        user_images_len = len(user_images)

        return render_template('dashboard/index.html', username=current_user.name, user_images=user_images, user_images_len=user_images_len)

    @app.route('/images/insertion/<filename>')
    @login_required
    def get_image(filename):
        return send_from_directory(app.config['MARKED_IMAGES_PATH'], filename)

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

            if watermark_file:
                watermark_file = cv2.imdecode(np.fromstring(
                    watermark_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            if file and allowed_file(file.filename):
                user_filename = f"{current_user.id}-{str(uuid.uuid4())}.png"

                file_path = os.path.join(
                    app.config['MARKED_IMAGES_PATH'], user_filename)
                file.save(file_path)

                original_image = cv2.imread(file_path)

                marked_image = perform_insertion(
                    original_image, insertion_type, 0.1, watermark_file, watermark_text)

                if marked_image is not None:
                    cv2.imwrite(file_path, marked_image)

                    UserService.add_image_to_user(
                        current_user.id, user_filename)

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
