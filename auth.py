from flask import session, redirect, url_for

# Fake user for demo purposes
USERS = {'admin': '1234'}

def login_required(func):
    def wrapper(*args, **kwargs):
        if 'username' in session:
            return func(*args, **kwargs)
        return redirect(url_for('login'))
    wrapper.__name__ = func.__name__
    return wrapper