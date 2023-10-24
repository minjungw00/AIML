import numpy as np
import sys
import os
import subprocess
import time
import msvcrt
import pyautogui

with subprocess.Popen(args=["cmd", "/k", 'python', '-u', "C:/Users/min/Desktop/Artech/3_2/AIML/bdc-client/submit.py", '--config', 'C:/Users/min/Desktop/Artech/3_2/AIML/assignment/assignment_01/config.yml'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as proc:

    # 파일 디스크립터를 가져옴
    fd = proc.stdout.fileno()

    # 파일 디스크립터를 상속 가능하도록 설정 (파이썬 3.4 이상)
    os.set_inheritable(fd, True)

    # 파일 디스크립터의 논블로킹 모드로 설정
    msvcrt.setmode(fd, os.O_BINARY)

    def read_output():
        try:
            last = ["", ""]

            while True:
                output = proc.stdout.read(1)  # 한 문자씩 읽음
                if output:
                    print(output, end='')

                    # 특정 출력에 대해 입력 전송 (예: "Select: ")
                    if output == " " and last[0] == "t" and last[1] == ":":
                        return
                    
                    last[0] = last[1]
                    last[1] = output
        except Exception as e:
            print(f"Error occurred: {e}")

    def init_submit():
        read_output()
        time.sleep(1)
        pyautogui.write('1')    # 1 키를 전송
        read_output()
        time.sleep(5)
        # 이때 비밀번호 입력할것
        read_output()
        time.sleep(1)

    def submit():
        time.sleep(2)
        pyautogui.write('1')
        time.sleep(3)
        read_output()

    def get_metric():
        f = open('C:/Users/min/Desktop/Artech/3_2/AIML/bdc-client/log.txt', 'r')
        lines = f.readlines()
        metric = (float)(lines[len(lines) - 5].split(' ')[2])
        f.close()
        return metric

    print(get_metric())

    # init_submit()

    # submit()

    proc.terminate()