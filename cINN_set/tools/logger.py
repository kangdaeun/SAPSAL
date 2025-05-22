#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:07:49 2025

@author: daeun

file for logger used in run_train, eval.,,,
"""


import sys

class Logger:
    def __init__(self, filename, log_mode=True, mode="a"):
        """
        로그를 파일에 저장하는 동시에 콘솔에 출력할 수 있는 클래스.
        
        :param filename: 로그를 저장할 파일 이름
        :param log_mode: True이면 콘솔과 파일 모두 출력, False이면 파일에만 출력
        :param mode: "w"는 새로운 파일 생성, "a"는 기존 파일에 이어쓰기
        """
        self.file = open(filename, mode)
        self.stdout = sys.stdout  # 원래 stdout 저장
        self.stderr = sys.stderr  # 원래 stderr 저장
        self.log_mode = log_mode

        # stdout, stderr 모두 대체
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        if self.log_mode:
            self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        if self.log_mode:
            self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout  # stdout 원복
        sys.stderr = self.stderr  # stderr 원복
