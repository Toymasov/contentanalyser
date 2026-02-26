
import re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    'page_icon=\"📰\",': '',
    '\"🔍 Filtrlash Sozlamalari\"': '\"Filtrlash Sozlamalari\"',
    '\"*(❗️ Yuklab olish uchun har bir grafikdagi Kamera 📷 tugmasiga bosing)*\"': '\"*(Yuklab olish uchun har bir grafikdagi Kamera tugmasiga bosing)*\"',
    '\"📰 Manba (Source)\"': '\"Manba (Source)\"',
    '\"🎭 Sarlavha Hissiyoti\"': '\"Sarlavha Hissiyoti\"',
    '\"⚠️ Zo\'ravonlik Holati\"': '\"Zo\'ravonlik Holati\"',
    '\"📑 Yangilik Turi\"': '\"Yangilik Turi\"',
    '\"📍 Mintaqa (Hudud)\"': '\"Mintaqa (Hudud)\"',
    '\"📊 Ko\'rsatilayotgan yangiliklar': '\"Ko\'rsatilayotgan yangiliklar',
    '\"📈 Manbalar bo\'yicha': '\"Manbalar bo\'yicha',
    '🗣️ ': '',
    '🌳 ': '',
    '🕸️ ': '',
    '☁️ ': '',
    '📄 ': '',
    '⚠️ ': '',
    '📥 ': '',
    '🎭 ': '',
    '📑 ': '',
    '📈 ': '',
    '💬 ': ''
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Also strip any stray emojis left by using simple ascii + cyrillic + standard punct match
# Actually just the targeted ones are safe so we don't break string encodings
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
