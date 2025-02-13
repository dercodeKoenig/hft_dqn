from ftplib import *


# FTP server details
ftp_server = 'benundmarvpromotions.lima-ftp.de'
ftp_user = 'benundmarvpromotions'
ftp_password = 'defrtrn2343fdsgfcdf'

model_save_name = "model.weights.h5"
file_to_upload = model_save_name
remote_path = '/test/' + model_save_name

# Connect to the FTP server
ftp = FTP(ftp_server)
ftp.login(user=ftp_user, passwd=ftp_password)

ftp.cwd("/")

# Open the file in binary mode and upload it
with open(file_to_upload, 'rb') as file:
    ftp.storbinary(f'STOR {remote_path}', file, blocksize = 1024*1024)

# Close the connection
ftp.quit()