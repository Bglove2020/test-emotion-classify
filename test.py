

#
import re


print(r"\\n")

#将中文转换为 Unicode 编码
chinese_str = "!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
for i in chinese_str:
    print(ord(i),end="")
    print(r"\u00",end="")
#b'\\u4e2d\\u6587'
print('我'.encode('utf8'))
str = b'abcd\xe6\x88\x91'
print(str.decode('utf8'))
pattern = re.compile(r'[^\u4e00-\u9fa5]+')
result = pattern.search('今晚真系食咗大头菜dawdawdawd')

print(result.group())
if result==None:
    print("YES")
print(u"U+27339")
print(hex(ord("」")))
print(re.compile(r'[\u4e00-\u9fa5?\']+').match("我的娃?'????").group())
print("wdawwadawdaw")

import torch

# 示例的输出和标签（可以根据你的实际情况进行替换）
output = torch.tensor([[0.1, 0.2, 0.7],
                       [0.8, 0.1, 0.1],
                       [0.3, 0.5, 0.2]])
labels = torch.tensor([2, 0, 1])

# 找到每个样本的预测类别
predicted_classes = torch.argmax(output, dim=1)

# 找到每个样本预测的概率
predicted_probabilities = output[torch.arange(output.size(0)), predicted_classes]

# 找到预测是否正确的布尔值
correct_predictions = predicted_classes == labels

# 打印结果
print("预测类别:", predicted_classes)
print("每个样本的预测概率:", predicted_probabilities)
print("每个样本的真实标签:", labels)
print("每个样本的预测是否正确:", correct_predictions)
