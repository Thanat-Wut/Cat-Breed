import pandas as pd

df = pd.read_csv('BU Data from Survey Cases_final(5).csv', encoding='cp874', header=1)

with open('mapping_analysis.txt', 'w', encoding='utf-8') as f:
    # The key insight: Thai characters are stored as '?' (0x3F) in the file
    # But we can still identify values by their string length and frequency patterns
    
    # For cols 5-9 (Purchase factors), the prompt says these should be:
    # มากที่สุด, มาก, ปานกลาง, น้อย, น้อยที่สุด
    # In Thai: มากที่สุด=9 chars, มาก=3 chars, ปานกลาง=7 chars, น้อย=4 chars, น้อยที่สุด=10 chars
    # Matching: len=3→มาก, len=4→น้อย, len=7→ปานกลาง, len=9→มากที่สุด, len=10→น้อยที่สุด
    
    f.write("=== PURCHASE FACTOR LIKERT MAPPING (cols 5-9) ===\n")
    f.write("Thai text → len → Likert value:\n")
    f.write("  มาก (3 chars) → len=3 '???' → Score 4\n")
    f.write("  น้อย (4 chars) → len=4 '????' → Score 2\n")
    f.write("  ปานกลาง (7 chars) → len=7 '???????' → Score 3\n")
    f.write("  มากที่สุด (9 chars) → len=9 '?????????' → Score 5\n")
    f.write("  น้อยที่สุด (10 chars) → len=10 '??????????' → Score 1\n\n")
    
    # For cols 22-71 (Option ratings), prompt says:
    # เห็นด้วยที่สุด, เห็นด้วย, เฉยๆ, ไม่เห็นด้วย, ไม่เห็นด้วยเลย
    # BUT we only see 4 unique values in cols 22-26!
    # เฉยๆ=4 chars, เห็นด้วย=8 chars, ไม่เห็นด้วย=11 chars, เห็นด้วยที่สุด=14 chars
    # Wait - ไม่เห็นด้วยเลย = 15 chars
    # Let me reconsider:
    # เฉยๆ = 4 chars → len=4
    # เห็นด้วย = 8 chars  → NOT matching (เ+ห+็+น+ด+้+ว+ย = 8 chars)
    # Hmm but the ? chars... Each Thai char could map to 1 ? 
    # Actually the file literally has '?' characters (0x3F bytes)
    # So len=4 means 4 question marks = 4 Thai chars originally
    
    # Thai character count:
    # เฉยๆ = เ,ฉ,ย,ๆ = 4 chars → len=4 '????'
    # เห็นด้วย = เ,ห,็,น,ด,้,ว,ย = 8 chars → len=8 '????????'
    # ไม่เห็นด้วย = ไ,ม,่,เ,ห,็,น,ด,้,ว,ย = 11 chars → len=11 '???????????'
    # เห็นด้วยที่สุด = เ,ห,็,น,ด,้,ว,ย,ท,ี,่,ส,ุ,ด = 14 chars → len=14 '??????????????'
    # ไม่เห็นด้วยเลย = ไ,ม,่,เ,ห,็,น,ด,้,ว,ย,เ,ล,ย = 14 chars → len=14 ALSO!
    # WAIT that's 14 too... but we only see 4 unique values
    # So maybe the survey only had 4 options, not 5
    
    f.write("=== OPTION RATING LIKERT MAPPING (cols 22-71) ===\n")
    f.write("Only 4 unique values found (not 5 as expected):\n")
    f.write("  เฉยๆ (4 chars) → len=4 '????' → Score 2\n")
    f.write("  เห็นด้วย (8 chars) → len=8 '????????' → Score 3\n")
    f.write("  ไม่เห็นด้วย (11 chars) → len=11 '???????????' → Score 1\n")
    f.write("  เห็นด้วยที่สุด (14 chars) → len=14 '??????????????' → Score 4\n\n")
    
    # Verify with col 22 frequency distribution
    f.write("=== Verification: Col 22 value distribution ===\n")
    vals = df.iloc[:, 22].dropna().value_counts()
    for v, c in vals.items():
        f.write(f"  '{v}' (len={len(v)}): {c} occurrences\n")
    
    # Demographics mapping
    f.write("\n=== DEMOGRAPHICS MAPPING ===\n")
    f.write("Col 73 (Age):\n")
    vals73 = df.iloc[:, 73].dropna().value_counts()
    for v, c in vals73.items():
        f.write(f"  '{v}' (len={len(v)}): {c}\n")
    
    f.write("\nCol 74 (Gender):\n")
    vals74 = df.iloc[:, 74].dropna().value_counts()
    for v, c in vals74.items():
        f.write(f"  '{v}' (len={len(v)}): {c}\n")
    # Thai: หญิง=5 chars, ชาย=3 chars, อื่นๆ=5 chars (but different)
    # Wait: ชาย = ช,า,ย = 3 chars → len=3
    # หญิง = ห,ญ,ิ,ง = 4 chars → len=4
    # ไม่ระบุ/LGBTQ+ etc = len=5
    
    f.write("\nCol 75 (Marital Status):\n")
    vals75 = df.iloc[:, 75].dropna().value_counts()
    for v, c in vals75.items():
        f.write(f"  '{v}' (len={len(v)}): {c}\n")

    # Col 1 experience
    f.write("\nCol 1 (Experience: เคย/ไม่เคย):\n")
    vals1 = df.iloc[:, 1].dropna().value_counts()
    for v, c in vals1.items():
        f.write(f"  '{v}' (len={len(v)}): {c}\n")
    # เคย = เ,ค,ย = 3 chars
    # ไม่เคย = ไ,ม,่,เ,ค,ย = 6 chars

print('Done')
