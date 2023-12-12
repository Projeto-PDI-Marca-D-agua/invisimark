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
ORIGINAL_IMAGES_PATH = os.path.join(USERS_IMAGES, 'original_images')
MARKED_IMAGES_PATH = os.path.join(USERS_IMAGES, 'marked_images')
WATERMARKS_PATH = os.path.join(USERS_IMAGES, 'watermarks')


def init_app(app):
    app.secret_key = 'super secret key'
    app.config['MARKED_IMAGES_PATH'] = MARKED_IMAGES_PATH
    app.config['ORIGINAL_IMAGES_PATH'] = ORIGINAL_IMAGES_PATH
    app.config['WATERMARKS_PATH'] = WATERMARKS_PATH
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

            flash('Credenciais inválidas.', 'danger')

        return render_template('auth/login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            email = request.form['email']
            name = request.form['name']
            password = request.form['password']

            result = UserService.register(email, name, password)

            if result == "Sucesso":
                flash('Usuário cadastrado com sucesso!', 'success')
                return redirect(url_for('login'))
            elif result == "E-mail em uso":
                flash('Este e-mail já está em uso.', 'danger')

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
        user_watermarks = current_user.watermarks
        user_images_len = len(user_images)
        user_watermarks_len = len(user_watermarks)

        return render_template('dashboard/index.html', username=current_user.name, email=current_user.email, user_images=user_images, user_images_len=user_images_len, user_watermarks_len=user_watermarks_len)

    @app.route('/images/insertion/<filename>')
    @login_required
    def get_image(filename):
        return send_from_directory(app.config['MARKED_IMAGES_PATH'], filename)

    @app.route('/images/watermark/<filename>')
    @login_required
    def get_watermark(filename):
        return send_from_directory(app.config['WATERMARKS_PATH'], filename)

    @app.route('/dashboard/insertion', methods=['GET', 'POST'])
    @login_required
    def insertion():
        user_watermarks = current_user.watermarks

        if request.method == 'POST':

            if 'image' not in request.files:
                flash('Nenhum arquivo enviado')
                return redirect(request.url)

            file = request.files['image']
            insertion_type = request.form['insertion_type']
            watermark_select = request.form['watermark_select']
            watermark_file = None
            watermark_text = None

            watermark_type = None
            for watermark in user_watermarks:
                if watermark['value'] == watermark_select:
                    watermark_type = watermark['type']
                    break

            if watermark_type == 'image':
                watermark_file = cv2.imdecode(np.fromstring(
                    open(os.path.join(app.config['WATERMARKS_PATH'], watermark_select), 'rb').read(), np.uint8), cv2.IMREAD_UNCHANGED)
            elif watermark_type == 'text':
                watermark_text = watermark_select
            if file.filename == '':
                flash('Nenhum arquivo selecionado')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                user_filename = f"{current_user.id}-{str(uuid.uuid4())}.png"

                file_path = os.path.join(
                    app.config['MARKED_IMAGES_PATH'], user_filename)
                file.save(file_path)

                original_image = cv2.imread(file_path)

                marked_image = perform_insertion(
                    original_image, insertion_type, watermark_file, watermark_text)

                if marked_image is not None:
                    cv2.imwrite(file_path, marked_image)

                    UserService.add_image_to_user(
                        current_user.id, user_filename)

                    return send_file(file_path, as_attachment=True)

            else:
                flash('Extensão de arquivo inválida')

            return redirect(url_for('dashboard'))

        return render_template('dashboard/insertion.html', username=current_user.name, email=current_user.email, user_watermarks=user_watermarks)

    @app.route('/dashboard/extraction', methods=['GET', 'POST'])
    @login_required
    def extraction():
        return render_template('/dashboard/extraction.html', username=current_user.name, email=current_user.email)

    @app.route('/dashboard/addwatermark', methods=['GET', 'POST'])
    @login_required
    def addwatermark():
        if request.method == 'POST':
            watermark_name = request.form['watermark_name']
            watermark_type = request.form['watermark_type']

            if watermark_type == 'text':
                watermark_value = request.form['watermark_text']
            elif watermark_type == 'image':
                if 'watermark_file' not in request.files:
                    flash('Nenhum arquivo enviado')
                    return redirect(request.url)

                watermark_file = request.files['watermark_file']

                if watermark_file.filename == '':
                    flash('Nenhum arquivo selecionado')
                    return redirect(request.url)

                filesave = save_watermark_file(watermark_file)
                watermark_value = os.path.basename(filesave)
            else:
                flash('Tipo de marca d\'água inválido')
                return redirect(request.url)

            try:
                UserService.add_watermark_to_user(
                    current_user.id, watermark_name, watermark_type, watermark_value)
                flash('Marca d\'água adicionada com sucesso!', 'success')
                return redirect(url_for('dashboard'))
            except ValueError as e:
                flash(str(e), 'danger')

        return render_template('dashboard/addwatermark.html', username=current_user.name, email=current_user.email)

    @app.route('/dashboard/mywatermarks')
    @login_required
    def watermarks():
        user_watermarks = current_user.watermarks
        return render_template('dashboard/mywatermarks.html', username=current_user.name, email=current_user.email, user_watermarks=user_watermarks)

    @app.route('/dashboard/myprofile')
    @login_required
    def myprofile():
        return render_template('dashboard/myprofile.html', username=current_user.name, email=current_user.email)

    @app.route('/dashboard/myimages')
    @login_required
    def myimages():
        return render_template('dashboard/myimages.html', username=current_user.name, email=current_user.email)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_insertion(original_image, insertion_type, watermark=None, text=None):
    if insertion_type == 'image_dct':
        marked_image = DCTImage.rgb_insert_dct(
            original_image, watermark)
    elif insertion_type == 'text_dct':
        marked_image = DCTText.rgb_insert_text_dct(original_image, text)
    elif insertion_type == 'image_dwt':
        marked_image = DWTImage.embed_watermark_HH_blocks(
            original_image, watermark)
    elif insertion_type == 'text_dwt':
        marked_image = DWTImage.embed_watermark_HH_blocks(
            original_image, watermark)
    else:
        flash('Tipo de inserção não suportado.')
        return None

    return marked_image


def save_watermark_file(file):
    watermark_filename = os.path.join(
        WATERMARKS_PATH, f"{str(uuid.uuid4())}.png")
    file.save(watermark_filename)
    return watermark_filename
