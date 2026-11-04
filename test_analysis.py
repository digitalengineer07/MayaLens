import urllib.request
import mimetypes
import uuid
import os
import json

def test_file_upload(file_path):
    url = 'http://127.0.0.1:8000/api/analyze'
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: {mime_type}\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"=== {filename} ===")
        print(f"Verdict: {res['decision']}")
        print(f"Confidence: {res['confidence']}%")
        print(f"Risk Level: {res['risk_level']}")
        print(f"Is Fake: {res['is_fake']}")
        for s in res['streams']:
            print(f"  * {s['name']}: {s['confidence']}% ({s['status']})")
        print()

print("1. Testing user human-edited video (Text_Animation):")
test_file_upload(r"D:\MayaLens\backend\uploads\Text_Animation-1786864110578_2.mp4")

print("2. Testing user deepfake video (No_still_this_is_s_vdo):")
test_file_upload(r"D:\MayaLens\backend\uploads\No_still_this_is_s_vdo_I_wan.mp4")

print("3. Testing graphic logo (mayalens_icon.png):")
test_file_upload(r"D:\MayaLens\frontend\public\mayalens_icon.png")
