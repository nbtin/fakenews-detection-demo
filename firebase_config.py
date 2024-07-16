# For Firebase JS SDK v7.20.0 and later, measurementId is optional
# firebaseConfig = {
#   "apiKey": "AIzaSyAjfaU_gONrfSCU1Yl50Z40AECMWhZbub4",
#   "authDomain": "fakenews-4048f.firebaseapp.com",
#   "projectId": "fakenews-4048f",
#   "storageBucket": "fakenews-4048f.appspot.com",
#   "messagingSenderId": "888362903257",
#   "appId": "1:888362903257:web:5d9cc77d9e5587597500c8",
#   "measurementId": "G-PBXJW8LNXT",
#   "serviceAccount": "service_account.json",
#   "databaseURL": "https://fakenews-4048f-default-rtdb.firebaseio.com"
# }

from firebase_admin import credentials, initialize_app, storage

# Init firebase with your credentials
cred = credentials.Certificate("service_account.json")
initialize_app(cred, {'storageBucket': 'fakenews-4048f.appspot.com'})

# Put your local file path 
fileName = "input_images/3.jpg"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)

# Opt : if you want to make public access from the URL
blob.make_public()

print("your file url", blob.public_url)
