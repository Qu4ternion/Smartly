# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:50:55 2022

@author: Acer
"""

path = r'C:\Users\Acer\Desktop\Bots\Chatbot\corpus\french\conversations.yml'

with open(path, 'r') as f:
    file = f.read()


data = file.split('\n')


for i in range(len(data)):
    data[i] = data[i].lstrip('- ')
    


write_path = r'C:\Users\Acer\Desktop\Smartly'
with open(path, 'w') as f:
    file = f.write(data)
    
    
