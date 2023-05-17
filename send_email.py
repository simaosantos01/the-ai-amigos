import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()


def send_email(best_fitness):
    email_message = MIMEMultipart()
    email_message['From'] = os.getenv('EMAIL')
    email_message['To'] = os.getenv('EMAIL')
    email_message['Subject'] = 'Genetic Algorithm Result'

    email_message.attach(MIMEText(str(f'best fitness: {best_fitness}'), 'plain'))

    smtp_server = smtplib.SMTP('smtp.office365.com', 587)
    smtp_server.starttls()
    smtp_server.login(os.getenv('EMAIL'), os.getenv('PASSWORD'))
    smtp_server.sendmail(os.getenv('EMAIL'), os.getenv('EMAIL'), email_message.as_string())
    smtp_server.quit()
