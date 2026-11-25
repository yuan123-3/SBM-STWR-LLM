import os
import requests
import zipfile
import io

url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
extract_dir = "ne_10m_admin_1_states_provinces"

print("Downloading map data...")
r = requests.get(url, timeout=120)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(extract_dir)
    
    # Move files from subdirectory to root
    sub_dir = os.path.join(extract_dir, extract_dir)
    if os.path.exists(sub_dir):
        for f in os.listdir(sub_dir):
            src = os.path.join(sub_dir, f)
            dst = os.path.join(extract_dir, f)
            os.rename(src, dst)
        os.rmdir(sub_dir)
    
    print("Done! Files:")
    for f in os.listdir(extract_dir):
        print(f"  {f}")
else:
    print("Download failed!")
