# user_service.py
import json
import uuid
import os
from flask_login import UserMixin, login_user


class UserService:
    REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    USERS_DATA = os.path.join(REPO_DIR, 'data/users.json')

    @staticmethod
    def load_user(user_id):
        users = UserService._load_users_from_file()
        for user_data in users:
            if user_data['id'] == user_id:
                user = UserService._create_user_instance(user_data)
                return user
        return None

    @staticmethod
    def register(email, name, password):
        users = UserService._load_users_from_file()

        for user_data in users:
            if user_data['email'] == email:
                return "Email already registered"

        new_user = {
            "id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password": password,
            "images": []
        }

        users.append(new_user)
        UserService._save_users_to_file(users)

        return "success"

    @staticmethod
    def _load_users_from_file():
        with open(UserService.USERS_DATA, 'r') as file:
            users = json.load(file)
        return users

    @staticmethod
    def _save_users_to_file(users):
        with open(UserService.USERS_DATA, 'w') as file:
            json.dump(users, file, indent=2)

    @staticmethod
    def _create_user_instance(user_data):
        user = User(user_data['id'], user_data['name'],
                    user_data['email'], user_data['password'])
        user.images = user_data['images']
        return user

    @staticmethod
    def authenticate_user(email, password):
        users = UserService._load_users_from_file()

        for user_data in users:
            if user_data['email'] == email and user_data['password'] == password:
                user = UserService._create_user_instance(user_data)
                login_user(user)
                return user

        return None

    @staticmethod
    def add_image_to_user(user_id, image_path):
        users = UserService._load_users_from_file()

        for user_data in users:
            if user_data['id'] == user_id:
                user_data['images'].append(image_path)
                UserService._save_users_to_file(users)
                return


class User(UserMixin):
    def __init__(self, user_id=None, name=None, email=None, password=None):
        self.id = user_id
        self.name = name
        self.email = email
        self.password = password
        self.images = []

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)
