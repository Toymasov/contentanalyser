import pandas as pd
import asyncio
import os
import sys

# Kutubxonani chaqiramiz
try:
    import gabriel
except ImportError:
    print("Xatolik: 'openai-gabriel' o'rnatilmagan! \nTerminaldan 'pip install openai-gabriel' buyrug'ini tushiring.")
    sys.exit(1)

async def main():
    input_csv = "analyzed_merged_news.csv"
    output_csv = "analyzed_merged_news.csv"
    
    # OpenAI TimeOut limitlarini kattalashtirish (Xatolik bermasligi uchun)
    os.environ["OPENAI_TIMEOUT"] = "120.0"
    os.environ["OPENAI_MAX_RETRIES"] = "5"
    
    # OS Environment ga API kalitni o'rnatish shart (Gabriel shunga qarab ish qiladi)
    # DIQQAT: API kalitni to'g'ridan to'g'ri kodga yozmang! Buni .env faylidan olishingiz kerak.
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "BU_YERGA_API_KALITINGIZNI_QOYMANG")
    
    print(f"[{input_csv}] O'qilmoqda...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Jadval o'qish xatosi: {e}")
        return

    # 1. Ma'lumotlarni moslashtirish
    # Odatda ko'plab qatorlar tahlillar uchun ulanayotganda bitta axborot qatlamiga aylanadi.
    df['Title'] = df.get('Title', '').fillna('')
    df['Content'] = df.get('Content', '').fillna('')
    
    # Qaysi joyda to'xtaganini eslab qolish kafolati uchun ID ni ko'rsatamiz
    if 'ID' in df.columns:
        df.set_index('ID', inplace=True)
    
    # 2. 'content length' ustunini qo'shish (belgilar soni)
    df['content length'] = df['Content'].astype(str).str.len()
    
    # Gabriel'ning "extract()" funksiyasi 1 ta kiruvchi maydon soraydi, shuning uchun 'Title' va 'Content' i 1 ta qatorga qo'shamiz
    df['Main_Text_For_AI'] = "Sarlavha: " + df['Title'].astype(str) + "\n\nMatn: " + df['Content'].astype(str)
    
    save_directory = os.path.abspath("gabriel_runs")
    
    # =========================================================================
    # YANGI BAZA KELGANDA ISHLATILADIGAN 1-BOSQICH (HOZIRCHA ARXIVGA OLINGAN)
    # Agar siz butunlay yangi "merged_news.csv" (boshlang'ich original holatdagi)
    # jadvaliga ulasangiz, qavslardagi (#) belgilarni olib tashlab ishlating.
    # =========================================================================
    # # 3. KENGAYTIRILGAN 'extract' Yo'riqnomalari (Gabriel Extraction qoidasi)
    # attributes = {
    #     "title sentiment analyse": "Sarlavha hissiyotini aniqla (Positive, Neutral, yoki Negative).",
    #     "content sentiment analyse": "Matnning umumiy hissiyotini aniqla (Positive, Neutral, yoki Negative).",
    #     "content analyse": "Matnning mazmuni haqida qisqacha tahlil (1-2 gapda).",
    #     "violence analyse": "Matnda umuman zo'ravonlik haqida ma'lumot bor yoki yo'qligi ('Bor' yoki 'Yo'q').",
    #     "violence_perpetrator": "Agar zo'ravonlik bo'lsa, zo'ravonlik qiluvchi (jinoyatchi, aybdor) shaxs kim ekanligini (masalan: er, ota, qo'shni, notanish shaxs) aniqla. Agar zo'ravonlik bo'lmasa 'Yo'q' deb yoz.",
    # }
    # 
    # print(f"\n1-BOSQICH: Asosiy faktlarni ajratib olish boshlanmoqda... \nJarayon '{save_directory}' ga yozib boriladi. Kutib turing...")
    # 
    # # 4. Gabriel 'extract' funksiyasi orqali faktlarni yig'ish
    # df = await gabriel.extract(
    #     df=df,
    #     column_name="Main_Text_For_AI",
    #     attributes=attributes,
    #     save_dir=os.path.join(save_directory, "extract_runs"),
    #     model="gpt-4o-mini",
    #     n_parallels=10, # API qotib qolishini umuman oldini olish uchun
    #     file_name="extraction_results.csv",
    #     modality="text",
    #     reset_files=False
    # )
    # =========================================================================

    print(f"\n[INFO] Yuborilgan '{input_csv}' da 1-BOSQICH qatorlari allaqachon tayyor deb topildi. \nFaqat zo'ravonlik 'Bor' bo'lgan qatorlar olib qolinib keyingi qadamlar boshlanmoqda...")
    
    # 5. Zo'ravonlik turi va Geografiyasi tasnifi (Faqat "Bor" qatorlar uchun maxsus)
    
    # [MUHIM YECHIM]: Jadvalda oldingi qolib ketgan ustunlar bilan aralashib "KeyError" ishlamay qolmasligi uchun eski ustunlarni tozalaymiz
    cols_to_drop = [
        'violence victim type', 'news type', 'violence location',
        'Ayollarga nisbatan', 'Bolalarga nisbatan', 'Ayollar va bolalarga nisbatan', 'Boshqa', "Zo'ravonlik holati yo'q",
        "O'zbekiston", "Xorij",
        'Qonun-qarorlar (Law)', 'Voqea-hodisa', 'Analitics/Tahliliy', 'Boshqa/Umumiy'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    print("\n2-BOSQICH: Zo'ravonlik holatlarining Ichki toifalarini o'rganish (Classify)...")
    if 'violence analyse' in df.columns:
        # P.S: User aytganidek Faqat "Bor" qatorlarni yuboramiz. Textdan qidirmasdan shundoq jadvaldagi ustunni o'zidan aniqlaymiz.
        mask_bor = df['violence analyse'].astype(str).str.lower().str.contains("bor")
        if mask_bor.any():
            df_violence = df[mask_bor].copy()
            # [MUHIM YECHIM - INDEX]: Gabriel classify funksiyasi ichkarida indexlarni nollashtirib (reset_index) 
            # yuborgani uchun, ularni 'Yo'q' qatorlar o'rniga yozib qo'yayotgan edi. Asl indexlarni asrab qolamiz.
            original_indices = df_violence.index
            
            # --- 5.1 QURBONLAR TOIFASI ---
            # Ushbu jadval faqat Bor deb topilgan jinoyatlarda ishlagani sababli "Yo'q" bandi olib tashlandi
            violence_labels = {
                "Ayollarga nisbatan": "Zo'ravonlik asosan ayollarga nisbatan ishlatilgan.",
                "Bolalarga nisbatan": "Zo'ravonlik asosan yosh bolalarga nisbatan ishlatilgan.",
                "Ayollar va bolalarga nisbatan": "Zo'ravonlik ham ayollarga, ham bolalarga nisbatan ishlatilgan.",
                "Boshqa": "Zo'ravonlik qurbonlari yuqoridagi toifalarga kirmaydi (masalan, erkaklar o'rtasidagi janjal, hayvonlarga nisbatan va h.k)."
            }
            
            df_violence_victim = await gabriel.classify(
                df=df_violence.copy(),
                column_name="Main_Text_For_AI",
                labels=violence_labels,
                save_dir=os.path.join(save_directory, "classify_violence"),
                model="gpt-4o-mini",
                n_parallels=10, 
                file_name="classify_violence.csv",
                reset_files=True,  # Oldingi tahlildan qolib ketgan "Yo'q" degan kesh xatosi uchun Tozalab yozish shart!
                additional_instructions="Faqat qat'iyan bitta toifani tanla, ikkita toifa bo'lmasin. Faqat bittasini tanlang."
            )
            
            if 'predicted_classes' in df_violence_victim.columns:
                df_violence_victim.rename(columns={'predicted_classes': 'violence victim type'}, inplace=True)
                df_violence['violence victim type'] = df_violence_victim['violence victim type'].values
                
            # --- 5.2 ZO'RAVONLIK GEOGRAFIYASI ---
            # [PROMPT YECHIMI]: AI chalg'imasligi uchun O'zbekiston/Xorij shartini juda qattiq qildik.
            geo_labels = {
                "O'zbekiston": "Faqat va faqat voqea O'zbekiston Respublikasi hududida yoki uning viloyatlarida (Toshkent, Samarqand va h.k) sodir bo'lganligi matnda ochiq aytilsagina shu toifani tanla.",
                "Xorij": "Voqea O'zbekistonga aloqador emas. Har qanday dunyo yangiliklari, xorijiy davlatlar (Rossiya, AQSH, qo'shni davlatlar) yoki umuman qayerdaligi aniq yozilmagan bo'lsa darhol faqat shu 'Xorij' toifasini tanla."
            }
            
            df_violence_geo = await gabriel.classify(
                df=df_violence.copy(),
                column_name="Main_Text_For_AI",
                labels=geo_labels,
                save_dir=os.path.join(save_directory, "classify_geo"),
                model="gpt-4o-mini",
                n_parallels=10, 
                file_name="classify_geo.csv",
                reset_files=True,
                additional_instructions="Diqqat: Matnni o'qi, agar O'ZBEKISTON dagi voqea ekanligi yaqqol yozilmagan bo'lsa, avtomatik ravishda 'Xorij' ni tanla. Ikkita tanlov bo'lmasin. Faqat bittasini tanlang."
            )
            
            if 'predicted_classes' in df_violence_geo.columns:
                df_violence_geo.rename(columns={'predicted_classes': 'violence location'}, inplace=True)
                df_violence['violence location'] = df_violence_geo['violence location'].values
                
            # --- 5.3 YANGILIK TURLARI ---
            print("3-BOSQICH: Yangiliklarni contentga ko'ra tasniflash ham 'Bor' qatorlar uchun... ")
            news_labels = {
                "Qonun-qarorlar (Law)": "Matn qonunchilikdagi o'zgarishlar, sud qarorlari, prezident yoki hukumat farmonlari haqida.",
                "Voqea-hodisa": "Muayyan bo'lib o'tgan jinoyat, baxtsiz hodisa yoki kundalik xabarlar (kriminalistika, YTH).",
                "Analitics/Tahliliy": "Muammoni sabablari, statistikasi va ijtimoiy kelib chiqishi chuqur o'rganilgan tahliliy maqola.",
                "Boshqa/Umumiy": "Yuqoridagi toifalarga mos kelmaydigan umumiy xabar."
            }
            
            df_violence_news = await gabriel.classify(
                df=df_violence.copy(),
                column_name="Main_Text_For_AI",
                labels=news_labels,
                save_dir=os.path.join(save_directory, "classify_news"),
                model="gpt-4o-mini",
                n_parallels=10, 
                file_name="classify_news.csv",
                reset_files=True,
                additional_instructions="Faqat qat'iyan bitta toifani tanla, ikkita toifa bo'lmasin. Faqat bittasini tanlang."
            )
            
            if 'predicted_classes' in df_violence_news.columns:
                df_violence_news.rename(columns={'predicted_classes': 'news type'}, inplace=True)
                df_violence['news type'] = df_violence_news['news type'].values
                
            # Asl indexlarni joyiga qaytarib ulash
            df_violence.index = original_indices
            
            # Orqaga, asosiy DF ga birlashtiramiz (Faqatgina mavjud bo'lsa)
            merge_cols = []
            if 'violence victim type' in df_violence.columns: merge_cols.append('violence victim type')
            if 'violence location' in df_violence.columns: merge_cols.append('violence location')
            if 'news type' in df_violence.columns: merge_cols.append('news type')
            
            if merge_cols:
                df = df.merge(df_violence[merge_cols], left_index=True, right_index=True, how='left')
    
    # Keraksiz qo'shilgan ustunni olib tashlash va tayyor jadvalni chiqazish
    if "Main_Text_For_AI" in df.columns:
        df = df.drop(columns=["Main_Text_For_AI"])
        
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nBARCHA TAHLILLAR YAKUNLANDI! \nNatijalar {output_csv} ga eksport qilindi.")

if __name__ == "__main__":
    asyncio.run(main())
