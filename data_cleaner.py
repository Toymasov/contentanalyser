import pandas as pd
import ast

def process_victim(val):
    if pd.isna(val): return val
    
    # Matndan list formatiga o'g'irish
    if isinstance(val, str) and val.startswith('['):
        try:
            lst = ast.literal_eval(val)
        except:
            return val
    elif isinstance(val, list):
        lst = val
    else:
        return val
        
    if not isinstance(lst, list):
        return val
        
    if len(lst) > 1:
        # 1-shart: "Ayollar va bolalarga nisbatan" bor bo'lsa faqat shuni qoldirish
        if "Ayollar va bolalarga nisbatan" in lst:
            return "['Ayollar va bolalarga nisbatan']"
        
        # 2-shart: Ham "Ayollarga nisbatan", ham "Bolalarga nisbatan" qatnashgan bo'lsa 
        if "Ayollarga nisbatan" in lst and "Bolalarga nisbatan" in lst:
            return "['Ayollar va bolalarga nisbatan']"
            
        # 3-shart: "Boshqa" javobi boshqalar bilan kelganda "Boshqa" ni o'chirish
        if "Boshqa" in lst:
            lst.remove("Boshqa")
            
    # Qolgan barcha holatlarni qaytarish
    return str(lst)

def process_news(val):
    if pd.isna(val): return val
    
    # Matndan list formatiga o'g'irish
    if isinstance(val, str) and val.startswith('['):
        try:
            lst = ast.literal_eval(val)
        except:
            return val
    elif isinstance(val, list):
        lst = val
    else:
        return val

    if not isinstance(lst, list):
        return val
        
    if len(lst) > 1:
        # 4-shart: 'Qonun-qarorlar (Law)' va 'Voqea-hodisa' birga kelganda, faqat 'Voqea-hodisa' ni qoldirish
        if "Qonun-qarorlar (Law)" in lst and "Voqea-hodisa" in lst:
            lst.remove("Qonun-qarorlar (Law)")
            
    return str(lst)

def main():
    file_path = "analyzed_merged_news.csv"
    print(f"[{file_path}] O'qilmoqda...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Xatolik: {e}")
        return
    
    # Shartlarni amaliyotga tadbiq qilish
    if 'violence victim type' in df.columns:
        df['violence victim type'] = df['violence victim type'].apply(process_victim)
        
    if 'news type' in df.columns:
        df['news type'] = df['news type'].apply(process_news)
        
    # Natijani saqlash
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print("Muvaffaqiyatli tozalandi va saqlandi!")

if __name__ == "__main__":
    main()
