import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ATTACH_FILE = BASE_DIR / "merged_output" / "strong_surge_pullback_final.csv"

SMTP_HOST = os.environ["SMTP_HOST"]
SMTP_PORT = int(os.environ["SMTP_PORT"])
SMTP_USER = os.environ["SMTP_USER"]
SMTP_PASS = os.environ["SMTP_PASS"]
EMAIL_TO = os.environ["EMAIL_TO"]

def main():
    if not ATTACH_FILE.exists():
        raise FileNotFoundError(f"未找到结果文件: {ATTACH_FILE}")

    msg = EmailMessage()
    msg["Subject"] = "强势冲高回调选股结果"
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.set_content("附件是本次运行生成的 strong_surge_pullback_final.csv，请查收。")

    with open(ATTACH_FILE, "rb") as f:
        file_data = f.read()

    msg.add_attachment(
        file_data,
        maintype="text",
        subtype="csv",
        filename="strong_surge_pullback_final.csv"
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

    print("邮件发送成功")

if __name__ == "__main__":
    main()