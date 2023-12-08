from flask import render_template, request, redirect, url_for, flash, send_file, send_from_directory
from invisimark.services.dct_image import DCTImage
from invisimark.services.dct_text import DCTText
from invisimark.services.dwt_image import DWTImage
from werkzeug.utils import secure_filename
import os
import cv2

users = [
    {"email": "user1@example.com", "password": "password1"},
    {"email": "user2@example.com", "password": "password2"},
]

UPLOAD_FOLDER = 'D:\Repositories\invisimark\images\insertion'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def init_app(app):

    app.secret_key = 'super secret key'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
                    return redirect(url_for('dashboard'))

            return "Credenciais inválidas"

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
    def dashboard():
        return render_template('dashboard/index.html')

    @app.route('/dashboard/insertion', methods=['GET','POST'])
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

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                original_image = cv2.imread(file_path)

                marked_image = perform_insertion(original_image, insertion_type, 0.1, watermark_file, watermark_text)

                if marked_image is not None:
                    cv2.imwrite(file_path, marked_image)
                    return send_file(file_path, as_attachment=True)

            else:
                flash('Extensão de arquivo inválida')

            return redirect(url_for('dashboard'))

        return render_template('dashboard/insertion.html')
    
    @app.route('/dashboard/myprofile')
    def myprofile():
        return render_template('dashboard/myprofile.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_insertion(original_image, insertion_type, alpha=0.1, watermark=None, text=None):
    if insertion_type == 'image_dct':
        marked_image = DCTImage.rgb_insert_dct_blocks(original_image, watermark, alpha)
    elif insertion_type == 'image_dwt':
        marked_image = DWTImage.embed_watermark_HH_blocks(original_image, watermark, alpha)
    elif insertion_type == 'text_dct':
        marked_image = DCTText.rgb_insert_texto(original_image, text, alpha)
    else:
        flash('Tipo de inserção não suportado')
        return None

    return marked_image