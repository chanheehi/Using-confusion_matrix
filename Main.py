import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch

# 타겟과 예측값 배열 생성
target = [
    'apple', 'apple', 'apple', 'apple',
    'banana', 'banana', 'banana', 'banana',
    'melon', 'melon', 'melon', 'melon',
    'strawberry', 'strawberry', 'strawberry', 'strawberry'
    ]
predict = [
    'apple', 'banana', 'melon', 'apple',
    'banana', 'banana', 'banana', 'banana',
    'melon', 'melon', 'strawberry', 'banana',
    'melon', 'apple', 'melon', 'melon'
    ]
labels = ['apple', 'banana', 'melon', 'strawberry']

# confusion matrix 함수에 타겟값과 예측값 넣고 반환값을 confusion_matrix에 저장
con_matrix = confusion_matrix(target, predict)  # (정답, 예측)

cm_display = ConfusionMatrixDisplay(confusion_matrix = con_matrix, display_labels = labels)
cm_display.plot()
plt.savefig(f'./confusion_matrix.png')
plt.close()

# 타겟값과 예측값을 숫자로 변환하기 위한 딕셔너리 생성
labels_dic = {
    'apple' : 0,
    'banana' : 1,
    'melon' : 2,
    'strawberry' : 3, 
}

# 타겟값과 예측값을 숫자로 변환
target_num, predict_num = [], []
for i, j in zip(target, predict):   # zip() 함수를 이용해 target과 predict를 동시에 반복
    target_num.append(labels_dic[i])
    predict_num.append(labels_dic[j])

# 배열을 numpy로 변환
target_num = np.array(target_num)
predict_num = np.array(predict_num)

# confusion matrix 함수에 타겟과 예측값 넣고 반환값을 confusion_matrix에 저장
con_matrix = confusion_matrix(target_num, predict_num)  # (정답, 예측)

cm_display = ConfusionMatrixDisplay(confusion_matrix = con_matrix, display_labels = ['apple', 'banana', 'melon', 'strawberry'])
cm_display.plot()
plt.savefig(f'./num_confusion_matrix.png')
plt.close()

# numpy를 torch로 변환
target_tensor = torch.Tensor(target_num)
predict_tensor = torch.Tensor(predict_num)

# confusion matrix 함수에 타겟과 예측값 넣고 반환값을 confusion_matrix에 저장
con_matrix = confusion_matrix(target_tensor, predict_tensor)  # (정답, 예측)

cm_display = ConfusionMatrixDisplay(confusion_matrix = con_matrix, display_labels = ['apple', 'banana', 'melon', 'strawberry'])
cm_display.plot()
plt.savefig(f'./torch_confusion_matrix.png')
plt.close()
