# SpatialMind：VSR Zero-shot 人工复核报告

## 范围与方法

项目作者逐张查看 30 条抽样图片，填写了 `manual_error_type` 和 `manual_notes`。样本来自 Zero-shot test 的 Base／LoRA 配对预测，按“LoRA 相对官方标签改对、改错、两者都错”各抽 10 条，并尽量覆盖项目自定义关系类别。这是定向抽样，不能估计全部 1,222 条测试样本的错误类型比例。

下表保留项目作者的原始人工判断和备注。`官方标签`、`Base`、`LoRA` 是原始数据与模型结果；`分组`仅按官方标签计算。人工意见与官方标签不同时，只记录分歧，不修改数据或重新计算模型指标。原始案例清单保存在本机 `output/qwen35_zeroshot_error_cases_30.json`，该目录不纳入 Git。

## 完整测试集指标

| 模型 | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 77.00% | 84.52% | 67.73% | 75.20% | 426 | 515 | 78 | 203 |
| LoRA（checkpoint-800） | 83.80% | 80.06% | 91.26% | 85.29% | 574 | 450 | 143 | 55 |

真实 False 样本中，LoRA 新增 87 条假阳性并纠正 22 条，假阳性净增 65 条。这解释了 Precision 下降；真实 True 样本中，纠正 150 条假阴性并新增 2 条。这些计数来自完整测试集，与下面 30 条人工抽样分开。

## 30 条人工记录

按人工备注开头的 True／False 统计，23 条与官方标签一致，7 条不同（index 140、1050、1143、1101、606、99、141）。这表示个人复核意见与数据标签的分歧，不等于已证明官方标注错误。index 474、1197 的备注已由填写者更正；下表采用更正后的原文。

| 分组 | index | 官方 / Base / LoRA | 人工判断 | 人工填写的错误类型 | 人工备注（原文） |
|---|---:|---|---|---|---|
| 改对 | 923 | False / True / False | False | 空间关系混淆：by 与 on | 人工判断为 False：猫站在马桶圈上，而不是在马桶旁边。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 改对 | 1104 | False / True / False | False | 空间关系混淆：contain 与 on | 人工判断为 False：猫躺在车上，而不是在车里面。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 改对 | 255 | True / False / True | True | 空间关系混淆：far away from 与 near | 人工判断为 True：猫离车很远。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 371 | True / False / True | True | 空间关系混淆：in front of 与 behind | 人工判断为 True：床在杯子前面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 193 | True / False / True | True | 空间关系混淆：across from 与 near | 人工判断为 True：行李箱在猫对面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 474 | False / True / False | False | 空间关系混淆：at the right side of 与 on | 人工判断为 False：遥控器在书的上方。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 改对 | 219 | True / False / True | True | 空间关系混淆：facing 与背对 | 人工判断为 True：猫面向键盘。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 261 | True / False / True | True | 空间关系混淆：below 与 above | 人工判断为 True：床在行李箱下面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 164 | True / False / True | True | 空间关系混淆：touching 与 near | 人工判断为 True：床与行李箱接触。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改对 | 69 | True / False / True | True | 空间关系混淆：into 与 on | 人工判断为 True：猫在马桶里面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 改错 | 1197 | False / False / True | False | 空间关系混淆：next to 与 on | 人工判断为 False：猫在床上。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 764 | False / False / True | False | 空间关系混淆：outside 与 inside | 人工判断为 False：猫在马桶里。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 140 | False / False / True | True | 空间关系混淆：far away from 与 near | 人工判断为 True：背包离车很远。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |
| 改错 | 1050 | True / True / False | False | 空间关系混淆：at the back of 与 on | 人工判断为 False：床在猫上。Base 预测 True 与图像及官方标签符合，但判断错误；LoRA 预测 False，判断正确。 |
| 改错 | 173 | False / False / True | False | 空间关系混淆：at the edge of 与 in the middle of | 人工判断为 False：遥控器在床的边缘。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 109 | False / False / True | False | 空间关系混淆：at the left side of 与 above | 人工判断为 False：手机在键盘上方。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 1040 | False / False / True | False | 空间关系混淆：facing 与 on | 人工判断为 False：猫在水槽上。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 1012 | False / False / True | False | 空间关系混淆：on top of 与 under | 人工判断为 False：猫在键盘上面。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 240 | False / False / True | False | 空间关系混淆：touching 与 near | 人工判断为 False：猫没有接触胡萝卜。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 改错 | 654 | False / False / True | False | 空间关系混淆：into 与 on | 人工判断为 False：猫在马桶上。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 都错 | 273 | False / True / True | False | 空间关系混淆：at the side of 与 on | 人工判断为 False：背包在床上。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断错误。 |
| 都错 | 1143 | False / True / True | True | 空间关系混淆：in 与 on | 人工判断为 True：猫在水槽里。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 都错 | 1101 | False / True / True | True | 空间关系混淆：near 与 on | 人工判断为 True：车在停车计时器附近。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 都错 | 606 | True / False / False | False | 空间关系混淆：at the back of 与 on | 人工判断为 False：猫在马桶上。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 都错 | 817 | True / False / False | True | 空间关系混淆：at the edge of 与 in the middle of | 人工判断为 True：猫在床的边缘。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断错误。 |
| 都错 | 91 | True / False / False | True | 空间关系混淆：left of 与 on | 人工判断为 True：猫在键盘左边。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断错误。 |
| 都错 | 99 | True / False / False | False | 空间关系混淆：facing away from 与 on | 人工判断为 False：猫背对水槽。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 都错 | 810 | False / True / True | False | 空间关系混淆：on 与 under | 人工判断为 False：猫在背包下。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断错误。 |
| 都错 | 141 | False / True / True | True | 空间关系混淆：adjacent to 与 far away from | 人工判断为 True：马桶在洗手池旁边。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 都错 | 241 | False / True / True | False | 空间关系混淆：in 与 on | 人工判断为 False：猫在车上。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断错误。 |

## 结论与待确认项

- 完整测试集上，LoRA 的 Accuracy 和 Recall 提高，Precision 降低。精确率变化由假阳性净增 65 条支持；30 条定向样本不能单独解释这种变化的训练机制。
- 人工记录指出“在物体内／在物体上”、前后与左右、接触与邻近、朝向、主客体顺序等关系值得关注。涉及人工与官方标签分歧的 7 条应先作为争议样本，不用于断言模型明确对错。
- 30 条记录是项目作者的单人复核意见；若要公开主张数据标签有误，宜再请第二人独立复核。
- Random 与 Zero-shot 使用不同训练集和测试集，不应把跨划分分数差异完全归因于是否见过物体概念。
