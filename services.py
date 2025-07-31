import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
import uuid

def send_otp_email(user_email, otp):
    sender_email = "prateeklonari@gmail.com"
    sender_password = "msia knzl jach msqz"
    receiver_email = user_email

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = "Your OTP Code"

    body = f"Your OTP code is: {otp}\nThis OTP is valid for 10 minutes."
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("OTP sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

# Helper function to generate a 12-character UUID
def generate_short_uuid():
    return str(uuid.uuid4()).replace('-', '')[:12]

# Helper function to generate a 6-digit OTP
def generate_otp():
    return str(random.randint(100000, 999999))
