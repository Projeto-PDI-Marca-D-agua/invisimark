from flask import render_template, request, redirect, url_for

users = [
    {"email": "user1@example.com", "password": "password1"},
    {"email": "user2@example.com", "password": "password2"},
]


def init_app(app):
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

    @app.route('/dashboard/insertion')
    def insertion():
        return render_template('dashboard/insertion.html')

    @app.route('/dashboard/myprofile')
    def myprofile():
        return render_template('dashboard/myprofile.html')
