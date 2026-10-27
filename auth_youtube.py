"""YouTube OAuth yetkilendirmesi yapar ve token.json dosyasını kaydeder."""
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

def main():
    secret = Path("client_secret.json")
    if not secret.exists():
        print("HATA: client_secret.json bulunamadı!")
        input("Çıkmak için Enter'a basın...")
        return

    print("Google oturum açma penceresi tarayıcınızda açılıyor...")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    try:
        creds = flow.run_local_server(
            port=0,
            open_browser=True,
            prompt="consent",
            login_hint="starlinkofmusk@gmail.com"
        )
    except Exception as e:
        print(f"\nHata oluştu: {e}")
        input("Çıkmak için Enter'a basın...")
        return

    Path("token.json").write_text(creds.to_json())
    print("\n" + "="*55)
    print("  TEBRİKLER! YouTube oturumu başarıyla açıldı!")
    print("  token.json kaydedildi. Artık yükleme yapabilirsiniz.")
    print("="*55)
    input("Kapatmak için Enter'a basın...")

if __name__ == "__main__":
    main()
