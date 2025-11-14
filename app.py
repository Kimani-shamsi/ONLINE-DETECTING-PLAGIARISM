from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from datetime import datetime
import re
from functools import wraps

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plagiarism_detector.db'
app.config['SESSION_PERMANENT'] = False  # Enforce login every time

db = SQLAlchemy(app)

# --------------------
# Load model and vectorizer
# --------------------
try:
    model = joblib.load('plagiarism_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✓ Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠ Warning: Could not load model files: {e}")
    print("  Please run training_model.py first!")
    model = None
    vectorizer = None

# --------------------
# DATABASE MODELS
# --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    submissions = db.relationship('Submission', backref='user', lazy=True)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    ai_score = db.Column(db.Float, nullable=False)
    human_score = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # 'AI' or 'Human'
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'ai_score': round(self.ai_score * 100, 2),
            'human_score': round(self.human_score * 100, 2),
            'prediction': self.prediction,
            'submitted_at': self.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
            'text_length': len(self.text)
        }

# --------------------
# LOGIN REQUIRED DECORATOR
# --------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --------------------
# TEXT PREPROCESSING + ANALYSIS
# --------------------
def clean_text(text: str) -> str:
    """Basic preprocessing: lowercase, remove extra whitespace."""
    if not isinstance(text, str):
        return ''
    t = text.lower()
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def analyze_text(text):
    """Analyze text using the trained model."""
    if model is None or vectorizer is None:
        return None, None, 'Error: Model not loaded'
    try:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        probs_by_class = dict(zip(classes, probabilities))
        human_score = float(probs_by_class.get('Human', probs_by_class.get(0, probabilities[0])))
        ai_score = float(probs_by_class.get('AI', probs_by_class.get(1, probabilities[1])))
        prediction = 'AI' if ai_score > human_score else 'Human'
        return ai_score, human_score, prediction
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return None, None, f'Error: {str(e)}'

# --------------------
# ROUTES
# --------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not all([username, email, password, confirm_password]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        if request.is_json:
            return jsonify({'success': True, 'message': 'Registration successful! Please login.'}), 201
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Always require login even if session existed
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            if request.is_json:
                return jsonify({'success': True, 'message': 'Login successful!'}), 200
            return redirect(url_for('dashboard'))

        if request.is_json:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()  # remove session completely
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user = db.session.get(User, session['user_id'])
    submissions = Submission.query.filter_by(user_id=session['user_id']).order_by(Submission.submitted_at.desc()).all()
    return render_template('dashboard.html', user=user, submissions=submissions)


@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()

        if not title or not text:
            error_msg = 'Title and text are required'
            if request.is_json:
                return jsonify({'success': False, 'message': error_msg}), 400
            return render_template('submission.html', error=error_msg)

        if len(text) < 50:
            error_msg = 'Text must be at least 50 characters'
            if request.is_json:
                return jsonify({'success': False, 'message': error_msg}), 400
            return render_template('submission.html', error=error_msg)

        ai_score, human_score, prediction = analyze_text(text)
        if ai_score is None:
            if request.is_json:
                return jsonify({'success': False, 'message': prediction}), 500
            return render_template('submission.html', error=prediction)

        submission = Submission(
            user_id=session['user_id'],
            title=title,
            text=text,
            ai_score=ai_score,
            human_score=human_score,
            prediction=prediction
        )

        try:
            db.session.add(submission)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            error_msg = f'Database error: {e}'
            if request.is_json:
                return jsonify({'success': False, 'message': error_msg}), 500
            return render_template('submission.html', error=error_msg)

        if request.is_json:
            return jsonify({
                'success': True,
                'message': 'Analysis complete!',
                'result': {
                    'id': submission.id,
                    'title': title,
                    'ai_score': round(ai_score * 100, 2),
                    'human_score': round(human_score * 100, 2),
                    'prediction': prediction
                }
            }), 200

        return redirect(url_for('reports'))

    return render_template('submission.html')


@app.route('/reports')
@login_required
def reports():
    submissions = Submission.query.filter_by(user_id=session['user_id']).order_by(Submission.submitted_at.desc()).all()
    return render_template('reports.html', submissions=submissions)


# --------------------
# API ENDPOINTS
# --------------------
@app.route('/api/submission/<int:submission_id>')
@login_required
def api_get_submission(submission_id):
    submission = db.session.get(Submission, submission_id)
    if not submission or submission.user_id != session['user_id']:
        return jsonify({'error': 'Submission not found'}), 404
    return jsonify(submission.to_dict()), 200


@app.route('/api/submissions')
@login_required
def api_get_submissions():
    submissions = Submission.query.filter_by(user_id=session['user_id']).order_by(Submission.submitted_at.desc()).all()
    return jsonify([s.to_dict() for s in submissions]), 200


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    return render_template('reset_password.html')


@app.route('/faq')
def faq():
    return render_template('FAQ.html')


# --------------------
# ERROR HANDLERS
# --------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# --------------------
# INITIALIZE DATABASE
# --------------------
def init_db():
    with app.app_context():
        db.create_all()
        print("✓ Database initialized successfully!")


# if __name__ == '__main__':
#     init_db()
#     app.run(debug=True, host='0.0.0.0', port=5000)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
