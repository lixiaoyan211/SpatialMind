# SpatialMind：VSR Random Test 错误案例人工复核

## 范围与判读规则

- 比较对象：Qwen3.5-2B Base 与 Random LoRA 的 checkpoint-1700；两者均在 VSR Random Test 的 2,195 条样本上评测。
- 案例来源：`D:\SpatialMind\output\qwen35_error_cases_30.json`。按官方标签定义的 improved、regressed、persistent_error 三组各抽取 10 条，共 30 条；这些是定性抽样，不能用来估计全测试集的错误比例。
- 下表的“官方标签 / Base / LoRA”均取自原始评测数据；“人工判断”与“人工复核记录”取自用户填写的案例 JSON。三组名称只描述相对于官方标签的预测对错，不代表人工复核后的对错。
- True／False 是图像与英文描述是否相符。人工判断与官方标签不同时标为“有分歧”；保留两种判断，不改动原始数据集标签或已发布的完整测试集指标。人工复核是单人定性判断，可能受关系定义、物体识别和画面遮挡影响。
- 原始模型回答保留在案例 JSON 中；这 30 条均没有输出格式错误。人工记录说明可见的判断依据，不能据此断言模型内部的错误原因。

## 按官方标签：Base 错、LoRA 对（10 条）

| 序号 | 图片 | 原描述 | 官方标签 / Base / LoRA | 人工判断 | 人工错误类型 | 人工复核记录 |
|---:|---|---|---|---|---|---|
| 1769 | `000000288753.jpg` | The person is touching the dining table. | true / false / true | true | 空间关系混淆：touching 与 near | 人工判断为 True：人接触着餐桌。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 573 | `000000354398.jpg` | The person is outside the car. | false / true / false | false | 空间关系混淆：outside 与 in | 人工判断为 False：人在车里。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 620 | `000000068743.jpg` | The donut is away from the dining table. | true / false / true | true | 空间关系混淆：away from 与 near | 人工判断为 True：甜甜圈离餐桌很远。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 1244 | `000000378214.jpg` | The sheep is behind the cow. | true / false / true | true | 空间关系混淆：behind 与 in front of | 人工判断为 True：羊在牛后面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 1661 | `000000540436.jpg` | The person is across from the sandwich. | true / false / true | true | 空间关系混淆：across from 与 near | 人工判断为 True：人在三明治对面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 721 | `000000303047.jpg` | The person is at the left side of the bed. | false / true / false | false | 空间关系混淆：at the left side of 与 on | 人工判断为 False：人在床的上。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |
| 375 | `000000263512.jpg` | The zebra is facing the person. | true / false / true | true | 空间关系混淆：facing 与背对 | 人工判断为 True：斑马面向人。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 1415 | `000000329551.jpg` | The cat is below the teddy bear. | true / false / true | true | 空间关系混淆：below 与 above | 人工判断为 True：猫在泰迪熊下面。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 347 | `000000309607.jpg` | The person is touching the donut. | true / false / true | true | 空间关系混淆：touching 与 near | 人工判断为 True：人在甜甜圈旁边。Base 预测 False 与图像及官方标签不符；LoRA 预测 True，判断正确。 |
| 1734 | `000000012315.jpg` | The cat is into the toilet. | false / true / false | false | 空间关系混淆：into 与 on | 人工判断为 False：猫在马桶上面。Base 预测 True 与图像及官方标签不符；LoRA 预测 False，判断正确。 |

## 按官方标签：Base 对、LoRA 错（10 条）

| 序号 | 图片 | 原描述 | 官方标签 / Base / LoRA | 人工判断 | 人工错误类型 | 人工复核记录 |
|---:|---|---|---|---|---|---|
| 1732 | `000000066355.jpg` | The teddy bear is touching the bed. | false / false / true | false | 空间关系混淆：touching 与 on | 人工判断为 False：泰迪熊在床上。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 1752 | `000000128113.jpg` | The couch is surrounding the person. | false / false / true | true（与官方标签有分歧） | 空间关系混淆：surrounding 与 near | 人工判断为 True：沙发在人的周围。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |
| 849 | `000000429811.jpg` | The sandwich is near the wine glass. | false / false / true | true（与官方标签有分歧） | 空间关系混淆：near 与 far away from | 人工判断为 True：三明治离酒杯很近。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |
| 518 | `000000238989.jpg` | The orange is in front of the banana. | false / false / true | false | 空间关系混淆：in front of 与 behind | 人工判断为 False：橙子在香蕉后面。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 1000 | `000000190156.jpg` | The cat is at the edge of the dining table. | false / false / true | true（与官方标签有分歧） | 空间关系混淆：at the edge of 与 on | 人工判断为 True：猫在餐桌边缘。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |
| 1454 | `000000134769.jpg` | The chair is at the left side of the pizza. | true / true / false | false（与官方标签有分歧） | 空间关系混淆：at the left side of 与 across | 人工判断为 False：椅子在披萨对面。Base 预测 True 与图像及官方标签符合，但判断错误；LoRA 预测 False，判断正确。 |
| 911 | `000000447741.jpg` | The cat is facing away from the cake. | false / false / true | false | 空间关系混淆：facing away from 与 facing | 人工判断为 False：猫面向蛋糕。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 310 | `000000468951.jpg` | The suitcase is under the cat. | false / false / true | true（与官方标签有分歧） | 空间关系混淆：under 与 on | 人工判断为 True：行李箱在猫下面。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |
| 202 | `000000522360.jpg` | The bed is touching the laptop. | false / false / true | false | 空间关系混淆：touching 与 near | 人工判断为 False：床与笔记本电脑接触。Base 预测 False 与图像及官方标签符合；LoRA 预测 True，判断错误。 |
| 792 | `000000546337.jpg` | The banana is inside the bowl. | false / false / true | true（与官方标签有分歧） | 空间关系混淆：inside 与 on | 人工判断为 True：香蕉在碗里面。Base 预测 False 与图像及官方标签符合，但判断错误；LoRA 预测 True，判断正确。 |

## 按官方标签：两者都错（10 条）

| 序号 | 图片 | 原描述 | 官方标签 / Base / LoRA | 人工判断 | 人工错误类型 | 人工复核记录 |
|---:|---|---|---|---|---|---|
| 353 | `000000569024.jpg` | The teddy bear is next to the mouse. | false / true / true | true（与官方标签有分歧） | 空间关系混淆：next to 与 near | 人工判断为 True：泰迪熊在鼠标旁边。Base 预测 True 与图像及官方标签不符，但判断正确；LoRA 预测 True，判断正确。 |
| 1419 | `000000066556.jpg` | The cat is in the middle of the bench. | true / false / false | true | 空间关系混淆：in the middle of 与 on | 人工判断为 True：猫在长椅中间。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断错误。 |
| 1341 | `000000174239.jpg` | The carrot is near the bowl. | true / false / false | false（与官方标签有分歧） | 空间关系混淆：near 与 far away from | 人工判断为 False：胡萝卜不在碗附近。Base 预测 False 与图像及官方标签不符，但判断正确；LoRA 预测 False，判断正确。 |
| 153 | `000000117774.jpg` | The cat is in front of the refrigerator. | true / false / false | true | 空间关系混淆：in front of 与 behind | 人工判断为 True：猫在冰箱前面。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断错误。 |
| 203 | `000000475967.jpg` | The donut is through the dog. | false / true / true | false | 空间关系混淆：through 与 on | 人工判断为 False：甜甜圈没穿过狗。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断错误。 |
| 679 | `000000106810.jpg` | The motorcycle is at the right side of the truck. | false / true / true | false | 空间关系混淆：at the right side of 与 at the left side of | 人工判断为 False：摩托车在卡车左边。Base 预测 True 与图像及官方标签不符；LoRA 预测 True，判断错误。 |
| 1760 | `000000514254.jpg` | The person is facing the bed. | true / false / false | false（与官方标签有分歧） | 空间关系混淆：facing 与 on | 人工判断为 false：人在床上。Base 预测 False 与图像及官方标签不符，但判断正确；LoRA 预测 False，判断正确。 |
| 885 | `000000108113.jpg` | The donut is on top of the dog. | true / false / false | false（与官方标签有分歧） | 空间关系混淆：on top of 与 near | 人工判断为 False：甜甜圈在狗旁边。Base 预测 False 与图像及官方标签不符，但判断正确；LoRA 预测 False，判断正确。 |
| 6 | `000000550405.jpg` | The bench is touching the dining table. | true / false / false | false（与官方标签有分歧） | 空间关系混淆：touching 与 near | 人工判断为 False：壁架与餐桌接触。Base 预测 False 与图像及官方标签不符，但判断正确；LoRA 预测 False，判断正确。 |
| 975 | `000000172135.jpg` | The cake is inside the wine glass. | true / false / false | true | 空间关系混淆：inside 与 on | 人工判断为 True：蛋糕在酒杯里面。Base 预测 False 与图像及官方标签不符；LoRA 预测 False，判断错误。 |

## 复核结果与解释边界

1. 30 条均已填写人工错误类型和说明。人工判断与官方标签有 11 条分歧：按官方标签划分的“LoRA 退步”组 6 条（1752、849、1000、1454、310、792）， “两者都错”组 5 条（353、1341、1760、885、6）；“LoRA 改善”组没有分歧。
2. 部分分歧会改变对单个案例的解释。例如，1752、849、1000、310、792 在官方标签下属于 LoRA 退步，而人工判断认为 LoRA 预测正确；1454 也由官方标签下的 LoRA 退步变为人工判断下的 LoRA 正确。这些变化仅适用于本次人工复核案例。
3. 接触、距离、包含、方向和朝向关系均出现边界或视角问题。个案分析应同时检查被比较物体是否存在、关系词的含义以及画面是否足以支持判断；人工填写的“错误类型”是观察归类，不是对模型推理机制的证明。
4. Random Test 的 Accuracy、Precision、Recall、F1 继续依据官方标签和全部 2,195 条样本报告。若要给出“按人工标签修正”的整体指标，需要独立制定复标协议并覆盖有代表性的样本；本报告不从 30 条定性样本外推整体指标。

