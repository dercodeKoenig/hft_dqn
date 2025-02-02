import os
import pymysql
from io import BytesIO
from ftplib import FTP
import random
import json
from tqdm import tqdm
import numpy as np
import io

import time
start_time = time.time()

# FTP details to upload the embeddings
FTP_HOST = 'benundmarvpromotions.lima-ftp.de'
FTP_USER = 'benundmarvpromotions'
FTP_PASS = 'gWEhrtjanrgy'
REMOTE_DIR = '/online_tts/data/'

# Database connection details
DB_HOST = "benundmarvpromotions.lima-db.de"
DB_USER = "USER433859_2345"
DB_PASS = "Sonne-12345"
DB_NAME = "db_433859_6"


lang_voice_dict = {
    "af_heart": "a", "af_alloy": "a", "af_aoede": "a", "af_bella": "a", "af_jessica": "a", "af_kore": "a", "af_nicole": "a", "af_nova": "a", "af_river": "a", "af_sarah": "a", "af_sky": "a",
    "am_adam": "a", "am_echo": "a", "am_eric": "a", "am_fenrir": "a", "am_liam": "a", "am_michael": "a", "am_onyx": "a", "am_puck": "a", "am_santa": "a",
    "bf_alice": "b", "bf_emma": "b", "bf_isabella": "b", "bf_lily": "b", "bm_daniel": "b", "bm_fable": "b", "bm_george": "b", "bm_lewis": "b",
    "jf_alpha": "j", "jf_gongitsune": "j", "jf_nezumi": "j", "jf_tebukuro": "j", "jm_kumo": "j",
    "zf_xiaobei": "z", "zf_xiaoni": "z", "zf_xiaoxiao": "z", "zf_xiaoyi": "z", "zm_yunjian": "z", "zm_yunxi": "z", "zm_yunxia": "z", "zm_yunyang": "z",
    "ef_dora": "e", "em_alex": "e", "em_santa": "e",
    "ff_siwis": "f",
    "hf_alpha": "h", "hf_beta": "h", "hm_omega": "h", "hm_psi": "h",
    "if_sara": "i", "im_nicola": "i",
    "pf_dora": "p", "pm_alex": "p", "pm_santa": "p"
}




# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf


pipelines = {}

def process(text, voice = "af_heart", speed = 1.1):

    lang = "a"
    if voice in lang_voice_dict:
        lang = lang_voice_dict[voice]
    print("using lang:", lang)

    if lang in pipelines:
        pipeline = pipelines[lang]
    else:
        pipe_t0 = time.time()
        pipeline = KPipeline(lang_code=lang) # <= make sure lang_code matches voice
        pipelines[lang] = pipeline
        print("created pipeline for", lang, "took", time.time()-pipe_t0, "seconds")
    
    
    generator = pipeline(
        text, voice=voice,
        speed=speed
    )
    
    all_audio = []  # List to store all audio data
    
    for i, (gs, ps, audio) in enumerate(generator):
        #print(i)  # Index
        print(gs) # Text
        print(ps) # Phonemes
        all_audio.append(audio)  # Collect audio segments
    
    # Concatenate all audio segments into one
    combined_audio = np.concatenate(all_audio)
    
    return combined_audio


pwr = 0

def truncate_after_n_newlines(text, n):
    parts = text.split("\n")  # Split by newlines
    if len(parts) > n:
        return "\n".join(parts[:n])  # Join only the first n parts
    return text  # Return original text if it has fewer than n newlines


while True:
    try:
        connection = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM `requests` WHERE output_path=''")
        data = [x for x in cursor.fetchall()]
        connection.close()
    except Exception as e:
        print(e)
        time.sleep(1)
        continue
    
    if(len(data) == 0):
        if(pwr == 1):
            os.system("powercfg /S faf11aa6-9bb4-4f50-98e1-d401f3368cfd")
            pwr = 0
        time.sleep(1)
    else:
        if(pwr == 0):
            os.system("powercfg /S 381b4222-f694-41f0-9685-ff5bb260df2e")
            pwr = 1
        
        try:
            user_text_total = {}
            
            for i in data:
                if i[7] not in user_text_total:
                    user_text_total[i[7]] = 0
            
                tlen = len(i[2])
                user_text_total[i[7]] += tlen
            
            next_id_to_process = sorted(user_text_total.items(), key=lambda item: item[1])[0][0]
            #print("")
            #print(user_text_total)
            #print("")
            for i in data:
                if i[7] == next_id_to_process:
                    data = i
                    break
                    
            print(data, len(data[2]))
    
            text = truncate_after_n_newlines(data[2], 50)
            voice = data[5]
            speed = data[6]
            speed = min(4,max(0.25,speed))

            process_t0 = time.time()
            
            audio = process(text, voice = voice, speed = speed)

            print("processing took", time.time()-process_t0, "seconds")
            # Save as a single file
             # Create a BytesIO buffer to hold the audio data in memory
            audio_buffer = io.BytesIO()
            
            # Write audio to the buffer instead of a file
            sf.write(audio_buffer, audio, 24000,  format='WAV')
            audio_buffer.seek(0)  # Rewind the buffer to the beginning
            
            # FTP upload
            output_name = data[0] + "_" + str(data[1]) + ".wav"
            
            ftp = FTP(FTP_HOST)
            ftp.login(FTP_USER, FTP_PASS)
            ftp.storbinary(f'STOR {REMOTE_DIR}/{output_name}', audio_buffer)
            ftp.quit()
                
        except Exception as e:
            print(e)
            output_name = "_error_"

        connection = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE `requests` SET output_path = %s, time = NOW() WHERE user = %s AND req_id = %s",
            (output_name, data[0], data[1])
        )
        connection.commit()
        connection.close()