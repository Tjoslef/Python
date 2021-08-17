#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:03:28 2021

@author: kubrt
"""

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
fig, ax = plt.subplots()
def player(dataset):
  dataset.type = dataset.type.map({'S':1, 'B':2})
  #print(aaron_judge.type)
  #print(aaron_judge['plate_x'])
  dataset = dataset.dropna(subset = ['plate_x', 'plate_z', 'type'])
  plt.scatter(x = dataset['plate_x'],y = dataset['plate_z'],c = dataset['type'],cmap = plt.cm.coolwarm,alpha = 0.25)
  training_set,validation_set = train_test_split(dataset,random_state = 1)
  classifier = SVC(kernel = 'rbf',gamma = 3,C = 1)
  classifier.fit(training_set[['plate_x','plate_z']],training_set['type'])
  draw_boundary(ax,classifier)
  print(classifier.score(training_set[['plate_x','plate_z']],training_set['type']))
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3) 
  plt.show()
player(aaron_judge)
player(jose_altuve)
player(david_ortiz)