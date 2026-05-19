import re

# Simulation of Row 345 extraction
test_row = {
    "ipc_section": "87",
    "bns_section": "",
    "title": "Kidnapping or abducting in order to murder or for ransom, etc.",
    "notes": "87 | Kidnapping or abducting in order to murder or for ransom, etc. | 366 | No change."
}

def test_on_row(row):
    col1 = str(row.get('ipc_section', '')).strip()
    col2 = str(row.get('bns_section', '')).strip()
    title = str(row.get('title', '')).strip()
    notes = str(row.get('notes', '')).strip()
    all_text = f" {col1} | {col2} | {title} | {notes} ".replace("\n", " ")
    
    print(f"DEBUG: All Text: {all_text}")
    
    ipc_key, bns_key = "", ""
    i_match = re.search(r"(?i)\bIPC\s*(\d+[A-Z]?)", all_text)
    b_match = re.search(r"(?i)\bBNS\s*(\d+[A-Z]?)", all_text)
    if i_match: ipc_key = i_match.group(1).upper()
    if b_match: bns_key = b_match.group(1).upper()
    
    if not ipc_key or not bns_key:
        if col1.isdigit() and not col2:
            nums = re.findall(r"\|\s*(\d{1,5})\s*\|", all_text)
            others = [n for n in nums if n != col1]
            if others: 
                print(f"DEBUG: Found DIFFERENT match in notes: {others[0]}")
                ipc_key, bns_key = others[0], col1
            else:
                 nums2 = [n for n in re.findall(r"\b(\d{1,5})\b", all_text) if n != col1]
                 if nums2: 
                    print(f"DEBUG: Found secondary match: {nums2[0]}")
                    ipc_key, bns_key = nums2[0], col1

    print(f"RESULT: IPC={ipc_key}, BNS={bns_key}")

if __name__ == "__main__":
    test_on_row(test_row)
