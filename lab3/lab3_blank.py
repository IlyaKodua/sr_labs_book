#!/usr/bin/env python
# coding: utf-8

# **Лабораторный практикум по курсу «Распознавание диктора», Университет ИТМО, 2021**		

# **Лабораторная работа №3. Построение дикторских моделей и их сравнение**
# 
# **Цель работы:** изучение процедуры построения дикторских моделей с использованием глубоких нейросетевых архитектур.
# 
# **Краткое описание:** в рамках настоящей лабораторной работы предлагается изучить и реализовать схему построения дикторских моделей с использованием глубокой нейросетевой архитектуры, построенной на основе ResNet-блоков. Процедуры обучения и тестирования предлагается рассмотреть по отношению к задаче идентификации на закрытом множестве, то есть для ситуации, когда дикторские классы являются строго заданными. Тестирование полученной системы предполагает использование доли правильных ответов (accuracy) в качестве целевой метрики оценки качества.
# 
# **Данные:** в качестве данных для выполнения лабораторной работы предлагается использовать базу [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).
# 
# **Содержание лабораторной работы**
# 
# 1. Подготовка данных для обучения и тестирования блока построения дикторских моделей.							
# 
# 2. Обучение параметров блока построения дикторских моделей без учёта процедуры аугментации данных.
# 
# 3. Обучение параметров блока построения дикторских моделей с учётом процедуры аугментации данных.
# 
# 4. Тестированное блока построения дикторских моделей.

# In[38]:


# IPython extension to reload modules before executing user code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import of modules
import os
import sys

sys.path.append(os.path.realpath('..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import download_dataset, concatenate, extract_dataset, part_extract, download_protocol, split_musan
from exercises_blank import train_dataset_loader, test_dataset_loader, ResNet, MainModel, train_network, test_network
from sam import SAM
from ResNetBlocks import BasicBlock
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters


# **1. Подготовка данных для обучения и тестирования детектора речевой активности**
# 
# В ходе выполнения лабораторной работы необходимы данные для выполнения процедуры обучения и процедуры тестирования нейросетевого блока генерации дикторских моделей. Возьмём в качестве этих данных звукозаписи, сохраненные в формат *wav*, из корпуса [VoxCeleb1 dev set](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html). Данный корпус содержит 148,642 звукозаписи (частота дискретизации равна 16кГц) для 1,211 дикторов женского и мужского пола, разговаривающих преимущественно на английском языке.
# 
# В рамках настоящего пункта требуется выполнить загрузку и распаковку звуковых wav-файлов из корпуса VoxCeleb1 dev set.
# 
# ![Рисунок 1](https://analyticsindiamag.com/wp-content/uploads/2020/12/image.png "VoxCeleb. Крупномасштабная аудиовизуальная база данных человеческой речи.")

# In[39]:


# # Download VoxCeleb1 (test set)
# with open('../data/lists/datasets.txt', 'r') as f:
#     lines = f.readlines()

# download_dataset(lines, user='voxceleb1902', password='nx0bl2v2', save_path='../data')


# In[40]:


# # Concatenate archives for VoxCeleb1 dev set
# with open('../data/lists/concat_arch.txt', 'r') as f:
#     lines = f.readlines()
    
# concatenate(lines, save_path='../data')


# In[41]:


# # Extract VoxCeleb1 dev set
# extract_dataset(save_path='../data/voxceleb1_dev', fname='../data/vox1_dev_wav.zip')


# In[42]:


# # Download VoxCeleb1 identification protocol
# with open('../data/lists/protocols.txt', 'r') as f:
#     lines = f.readlines()
    
# download_protocol(lines, save_path='../data/voxceleb1_test')


# **2. Обучение параметров блока построения дикторских моделей без учёта процедуры аугментации данных**
# 
# Построение современных дикторских моделей, как правило, выполняется с использованием нейросетевых архитектур, многие из которых позаимствованы из области обработки цифровых изображений. Одними из наиболее распространенных нейросетевых архитектур, используемыми для построения дикторских моделей, являются [ResNet-подобные архитектуры](https://arxiv.org/pdf/1512.03385.pdf). В рамках настоящего пункта предлагается выполнить адаптацию нейросетевой архитектуры ResNet34 для решения задачи генерации дикторских моделей (дикторских эмбеддингов). *Дикторский эмбеддинг* – это высокоуровневый вектор-признаков, состоящий, например, из 128, 256 и т.п. значений, содержащий особенности голоса конкретного человека. При решении задачи распознавания диктора можно выделить эталонные и тестовые дикторские эмбеддинги. *Эталонные эмбеддинги* формируются на этапе регистрации дикторской модели определённого человека и находятся в некотором хранилище данных. *Тестовые эмбеддинги* формируются на этапе непосредственного использования системы голосовой биометрии на практике, когда некоторый пользователь пытается получить доступ к соответствующим ресурсам. Система голосовой биометрии сравнивает по определённой метрике эталонные и тестовые эмбеддинги, формируя оценку сравнения, которая, после её обработки блоком принятия решения, позволяет сделать вывод о том, эмбеддинги одинаковых или разных дикторов сравниваются между собой.
# 
# Адаптация различных нейросетевых архитектур из обработки изображений к решению задачи построения дикторских моделей является непростой задачей. Возьмём за основу готовое решение, предложенной в рамках [следующей публикации](https://arxiv.org/pdf/2002.06033.pdf) и адаптируем его применительно к выполнению настоящей лабораторной работы.
# 
# Необходимо отметить, что построение дикторских моделей, как правило, требует наличия *акустических признаков*, вычисленных для звукозаписей тренировочной, валидационной и тестовой баз данных. В качестве примера подобных признаков в рамках настоящей лабораторной работы воспользуемся *логарифмами энергий на выходе мел-банка фильтров*. Важно отметить, что акустические признаки подвергаются некоторым процедурам предобработки перед их непосредственной передачей в блок построения дикторских моделей. В качестве этих процедур можно выделить: нормализация и масштабирование признаков, сохранение только речевых фреймов на основе разметки детектора речевой активности и т.п.
# 
# После того, как акустические признаки подготовлены, они могут быть переданы на блок построения дикторских моделей. Как правило, структура современных дикторских моделей соответствует структуре [x-векторных архитектур](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf). Эти архитектуры состоят из четырёх ключевых элементов: 
# 
# 1. **Фреймовый уровень.** Предназначен для формирования локальных представлений голоса конкретного человека. На этом уровне как раз и применяются нейросетевые архитектуры на базе свёрточных нейронных сетей, например, ResNet, позволяющих с использованием каскадной схемы из множества фильтров с локальной маской захватить некоторый локальный контекст шаблона голоса человека. Выходом фреймового уровня является набор высокоуровневых представлений (карт-признаков), содержащих локальные особенности голоса человека.
# 
# 2. **Уровень статистического пулинга** позволяет сформировать промежуточный вектор-признаков, фиксированной длины, которая является одинаковой для звукозаписи любой длительности. В ходе работы блока статистического пулинга происходит удаление временной размерности, присутствующей в картах-признаков. Это достигается путём выполнения процедуры усреднения карт-признаков вдоль оси времени. Выходом уровня статистического пулинга являются вектор среднего и вектор среднеквадратического отклонения, вычисленные на основе карт-признаков. Эти вектора конкатенируются и передаются для дальнейшей обработки на сегментом уровне.
# 
# 3. **Сегментный уровень.** Предназначен для трансформации промежуточного вектора, как правило, высокой размерности, в компактный вектор-признаков, представляющий собой дикторский эмбеддинг. Необходимо отметить, что на сегментном уровне расположены один или несколько полносвязных нейросетевых слоёв, а обработка данных выполняется по отношению ко всей звукозаписи, а не только к некоторому её локальному контексту, как на фреймовом уровне.
# 
# 4. **Уровень выходного слоя.** Представляет полносвязный слой с softmax-функциями активации. Количество активаций равно числу дикторов в тренирочной выборке. На вход выходноя слоя подаётся дикторский эмбеддинг, а на выходе – формируется набор апостериорных вероятностей, определяющих принадлежность эмбеддинга к одному из дикторских классов в тренировочной выборке. Необходимо отметить, что, как правило, в современных нейросетевых системах построения дикторских моделей выходной используется только на этапе обучения параметров и на этапе тестирования не используется (на этапе тестирования используются только три первых уровня архитектуры).
# 
# Обучение модели генерации дикторских эмбеддингов выполняется путём решения задачи *классификации* или, выражаясь терминами из области биометрии, *идентификации на закрытом множестве* (количество дикторских меток является строго фиксированным). В качестве используемой стоимостной функции выступает *категориальная кросс-энтропия*. Обучение выполняется с помощью мини-батчей, содержащих короткие фрагменты карт акустических признаков (длительностью несколько секунд) различных дикторов из тренировочной базы данных. Обучение на коротких фрагментов позволяет избежать сильного переобучения нейросетевой модели. При выполнении процедуры обучения требуется подобрать набор гиперпараметров, выбрать обучения и метод численной оптимизации.
# 
# Для успешного выполнения настоящего пункта необходимо сделать следующее:
# 
# 1. Сгенерировать списки тренировочных, валидационных и тестовых данных на основе идентификационного протокола базы VoxCeleb1, содержащегося в файле **../data/voxceleb1_test/iden_split.txt**. При генерации списков требуется исключить из них звукозаписи дикторов, которые входят в базу [VoxCeleb1 test set](https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip). Это позволит выполнить тестирования обученных блоков генерации дикторских моделей на протоколе [VoxCeleb1-O cleaned](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt), который составлен по отношению к данным из VoxCeleb1 test set, в лабораторной работе №4.
# 
# 2. Инициализировать обучаемую дикторскую модель, выбрав любой возможный вариант её архитектуры, предлагаемый в рамках лабораторной работы. При реализации блока статистического пулинга предлагается выбрать либо его классический вариант, предложенный в [следующей работе](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf), либо его более продвинутую версию основанную на использовании [механизмов внимания](https://arxiv.org/pdf/1803.10963.pdf). Использование последней версии статистического пулинга позволяет реализовать детектор речевой активности прямо внутри блока построения дикторских моделей.
# 
# 3. Инициализировать загрузчики тренировочной и валидационной выборки.
# 
# 4. Инициализировать оптимизатор и планировщик для выполнения процедуры обучения.
# 
# 5. Описать процедуру валидации/тестирования блока построения дикторских моделей.
# 
# 6. Описать процедуру обучения и запустить её, контролируя значения стоимостной функции и доли правильных ответов на тренировочном множестве, а также долю правильных ответов на валидационном множестве.

# In[43]:


# Select hyperparameters

# Acoustic features
n_mels            = 40                                   # number of mel filters in bank filters
log_input         = True                                 # logarithm of features by level

# Neural network archtecture
layers            = [3, 4, 6, 3]                         # number of ResNet blocks in different level of frame level
activation        = nn.ReLU                              # activation function used in ResNet blocks
num_filters       = [32, 64, 128, 256]                   # number of filters of ResNet blocks in different level of frame level
encoder_type      = 'SP'                                 # type of statistic pooling layer ('SP'  – classical statistic pooling 
                                                         # layer and 'ASP' – attentive statistic pooling)
nOut              = 512                                  # embedding size

# Loss function for angular losses
margin            = 0.35                                 # margin parameter
scale             = 32.0                                 # scale parameter

# Train dataloader
max_frames_train  = 200                                  # number of frame to train
train_path        = '../data/voxceleb1_dev/wav'          # path to train wav files
batch_size_train  = 128                                  # batch size to train
pin_memory        = False                                # pin memory
num_workers_train = 5                                    # number of workers to train
shuffle           = True                                 # shuffling of training examples

# Validation dataloader
max_frames_val    = 200                                 # number of frame to validate
val_path          = '../data/voxceleb1_dev/wav'          # path to val wav files
batch_size_val    = 32                                  # batch size to validate
num_workers_val   = 5                                    # number of workers to validate

# Test dataloader
max_frames_test   = 200                                 # number of frame to test
test_path         = '../data/voxceleb1_dev/wav'          # path to val wav files
batch_size_test   = 32                                  # batch size to test
num_workers_test  = 5                                    # number of workers to test

# Optimizer
lr                = 0.1                                  # learning rate value
weight_decay      = 0                                    # weight decay value

# Scheduler
val_interval      = 5                                    # frequency of validation step
max_epoch         = 40                                   # number of epoches

# Augmentation
musan_path        = '../data/musan_split'                # path to splitted SLR17 dataset
rir_path          = '../data/RIRS_NOISES/simulated_rirs' # path to SLR28 dataset


# In[44]:


# Generate data lists
train_list = []
val_list   = []
test_list  = []

with open('../data/voxceleb1_test/iden_split.txt', 'r') as f:
    lines = f.readlines()
    
black_list = os.listdir('../data/voxceleb1_test/wav')   # exclude speaker IDs from VoxCeleb1 test set
num_train_spk = []                                      # number of train speakers

for line in lines:
    line   = line.strip().split(' ')
    spk_id = line[1].split('/')[0]
    
    if not (spk_id in black_list):
        num_train_spk.append(spk_id)
        
    else:
        continue
    
    # Train list
    if (line[0] == '1'):
        train_list.append(' '.join([spk_id, line[1]]))
    
    # Validation list
    elif (line[0] == '2'):
        val_list.append(' '.join([spk_id, line[1]]))
    
    # Test list
    elif (line[0] == '3'):
        test_list.append(' '.join([spk_id, line[1]]))
        
num_train_spk = len(set(num_train_spk))


# In[45]:


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = ResNet(BasicBlock, layers=layers, activation=activation, num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, n_mels=n_mels, log_input=log_input)
trainfunc  = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
main_model = MainModel(model, trainfunc).to(device)


# In[46]:


# Initialize train dataloader (without augmentation)
train_dataset = train_dataset_loader(train_list=train_list, max_frames=max_frames_train, train_path=train_path)
train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=pin_memory, num_workers=num_workers_train, shuffle=shuffle)

# Initialize validation dataloader
val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, test_path=val_path)
val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val)


# In[47]:


# Initialize optimizer and scheduler

base_optimizer = torch.optim.SGD 
optimizer = SAM(main_model.parameters(),base_optimizer, lr=lr, weight_decay=weight_decay,  rho=0.05, adaptive=False)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200)
# scheduler = OneCycleLRScheduler(optimizer.base_optimizer, 
#                                 pct_start=0.30, 
#                                 cycle_momentum=False, 
#                                 max_lr=lr, 
#                                 div_factor=20, 
#                                 final_div_factor=10000, 
#                                 total_steps=max_epoch*len(train_loader))


# In[48]:


torch.cuda.is_available()


# In[49]:


start_epoch = 0
checkpoint_flag = False

if checkpoint_flag:
    start_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models/lab3_model_0004.pth')
    start_epoch = start_epoch + 1

# Train model
for num_epoch in range(start_epoch, max_epoch):
    train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=True)
    
    print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

    if (num_epoch + 1)%val_interval == 0:
        _, val_top1 = test_network(val_loader, main_model)
        
        print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1))
        
        saveParameters(main_model, optimizer, scheduler, num_epoch, path='../data/lab3_models')


# **3. Обучение параметров блока построения дикторских моделей с учётом процедуры аугментации данных**
# 
# Известно, что рроцедуры формирования и передачи речевого сигнала могут сопровождаться воздействием шумов и помех, приводящих к искажению сигнала. В качестве примеров искажающих факторов, влияющих на ухудшение качестве речевого сигнала можно привести: импульсный отклик помещения (реверберация), фоновый шум голосов группы нецелевых дикторов, звук телевизора или радиоприёмника и т.п. Разработка конвейера системы голосовой биометрии требует учёта воздействия искажающих факторов на качество её работы. Поскольку процедура построения современных дикторских моделей основана на обучении глубоких нейронных сетей, требующих большие объёмы данных для обучения их параметров, возможным вариантом увеличения тренировочной выборки может являться использование методов аугментации статистических данных. *Аугментация* – методика создания дополнительных обучающих примеров из имеющихся данных путём внесения в них искажений, которые могут потенциально возникнуть на этапе итогового тестирования системы.
# 
# Как правило, при решении задачи аугментации данных в речевой обработке используются дополнительные базы шумов и помех. В качестве примеров можно привести базы [SLR17](https://openslr.org/17/) (корпус музыкальных, речевых и шумовых звукозаписей) и [SLR28](https://openslr.org/28/) (база данных реальных и симулированных импульсных откликов комнат, а также изотропных и точечных шумов). Важно отметить, что перед применением с использованием методов аугментации подобных баз к имеющимся данным, требуется убедиться, что частоты дискретизации искажающих баз и оригинальных данных являются одинаковыми. Применительно к рассматриваемому лабораторному практикуму частоты дискретизации всех используемых звукозаписей должны быть равными 16кГц.
# 
# Как известно, можно выделить два режима аугментации данных: *онлайн* (применяется в ходе процедуры обучения) и *оффлайн* (применяется до процедуры обучения) аугментацию. В рамках настоящей лабораторной работы предлагается использовать онлайн аугментацию в силу не очень большого набора тренировочных данных и большей гибкости экспериментов, чем вс случае онлайн аугментации. Необходимо отметить, что применение онлайн аугментации на практике замедляет процедуру обучения, по сравнению с оффлайн аугментацией, так как наложение искажений, извлечение акустических признаков и их возможная предобработка требует определённого машинного времени.
# 
# В рамках настоящего пункта предлагается сделать следующее:
# 
# 1. Загрузить и извлечь данные из базы SLR17 (MUSAN). Частота дискретизации данных в рассматриваемой базе равна 16кГц по умолчанию. Поскольку звукозаписи рассматриваемой базы являются достаточно длинными, рекомендуется предварительно разбить эту базу на более маленькие фрагменты (например, длительностью 5 секунд с шагом 3 секунды), сохранив их на диск. 
# 
# 2. Загрузить и извлечь данные из базы SLR28 (MUSAN). Частота дискретизации данных в рассматриваемой базе равна 16кГц по умолчанию.
# 
# 3. Модернизировать загрузчик тренировочных данных под возможность случайного наложения (искажаем исходные звукозаписи) и не наложения (не искажаем исходные звукозаписи) одного из четырёх типов искажений (реверберация, музыкальный шум, фоновый шум голосов нескольких дикторов, неструктурированный шум), описанных внутри класса **AugmentWAV** следующего программного кода: **../common/DatasetLoader.py**.
# 
# 4. Используя процедуру обучения из предыдущего пункта с идентичными настройками выполнить тренировку параметров блока генерации дикторских моделей на исходных данных при наличии их аугментирвоанных копий.

# In[ ]:


# Download SLR17 (MUSAN) and SLR28 (RIR noises) datasets
with open('../data/lists/augment_datasets.txt', 'r') as f:
    lines = f.readlines()
    
download_dataset(lines, user=None, password=None, save_path='../data')


# In[ ]:


# Extract SLR17 (MUSAN)
extract_dataset(save_path='../data', fname='../data/musan.tar.gz')

# Extract SLR28 (RIR noises)
part_extract(save_path='../data', fname='../data/rirs_noises.zip', target=['RIRS_NOISES/simulated_rirs/mediumroom', 'RIRS_NOISES/simulated_rirs/smallroom'])


# In[ ]:


# Split MUSAN (SLR17) dataset for faster random access
split_musan(save_path='../data')


# In[ ]:


# Initialize model
model      = ResNet(BasicBlock, layers=layers, activation=activation, num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, n_mels=n_mels, log_input=log_input)
trainfunc  = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
main_model = MainModel(model, trainfunc).cuda()


# In[ ]:


# Initialize train dataloader (without augmentation)
train_dataset = train_dataset_loader(train_list=train_list, 
                                     max_frames=max_frames_train, 
                                     train_path=train_path, 
                                     augment=True, 
                                     musan_path=musan_path, 
                                     rir_path=rir_path)

train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=pin_memory, num_workers=num_workers_train, shuffle=shuffle)

# Initialize validation dataloader
val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, test_path=val_path)
val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val)


# In[ ]:


# Initialize optimizer and scheduler

base_optimizer = torch.optim.SGD 
optimizer = SAM(main_model.parameters(),base_optimizer, lr=lr, weight_decay=weight_decay,  rho=0.05, adaptive=False)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200)


# base_optimizer = 
# optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = OneCycleLRScheduler(optimizer, 
#                                 pct_start=0.30, 
#                                 cycle_momentum=False, 
#                                 max_lr=lr, 
#                                 div_factor=20, 
#                                 final_div_factor=10000, 
#                                 total_steps=max_epoch*len(train_loader))


# In[ ]:


start_epoch = 0
checkpoint_flag = False

if checkpoint_flag:
    start_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models_aug/lab3_model_0004.pth')
    start_epoch = start_epoch + 1

# Train model
for num_epoch in range(start_epoch, max_epoch):
    train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=True)
    
    print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

    if (num_epoch + 1)%val_interval == 0:
        _, val_top1 = test_network(val_loader, main_model)
        
        print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1))
        
        saveParameters(main_model, optimizer, scheduler, num_epoch, path='../data/lab3_models_aug')


# **4. Тестирование блока построения дикторских моделей**
# 
# Из литературы известно, что применение алгоритмов машинного обучения на практике требует использования трёх наборов данных: *тренировочное множество* (используется для обучения параметров модели), *валидационное множество* (используется для настройки гиперпараметров), *тестовое множество* (используется для итогового тестирования).
# 
# В рамках настоящего пункта предлагается выполнить итоговое тестирования блоков генерации дикторских моделей, обученных без аугментации и с аугментацией тренировочных данных, и сравнить полученные результаты. При проведении процедуры тестирования рекомендуется выбрать различное количество фреймов для тестовых звукозаписей, чтобы грубо понять то, как длительность фонограммы влияет на качество распознавания диктора.
# 
# В качестве целевой метрики предлагается использовать *долю правильных ответов*, то есть количество верно классифицированных объектов по отношению к общему количеству объектов тестового множества. Как и при проведении процедуры обучения и валидации, рассматриваемая процедура тестирования предполагает решение задачи идентификации диктора на закрытом множестве.

# In[ ]:


# Initialize test dataloader
test_dataset = test_dataset_loader(test_list=test_list, max_frames=max_frames_test, test_path=test_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=num_workers_test)


# In[ ]:


# Load model without augmentation
num_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models/lab3_model_0039.pth')

# Test model
_, test_top1 = test_network(test_loader, main_model)

print("Epoch {:1.0f}, Accuracy (test set) {:2.3f}%".format(num_epoch, test_top1))


# In[ ]:


# Load model with augmentation
num_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models_aug/lab3_model_0039.pth')

# Test model
_, test_top1 = test_network(test_loader, main_model)

print("Epoch {:1.0f}, Accuracy (test set) {:2.3f}%".format(num_epoch, test_top1))


# **5. Контрольные вопросы**
# 
# 1. Что такое верификация и идентицикация диктора?
# 
# 2. Что такое распознавание диктора на закрытом и открытом множестве?
# 
# 3. Что такое текстозависимое и текстонезависимое распознавание диктора?
# 
# 4. Описать схему обучения блока генерации дикторских моделей на основе нейронных сетей.
# 
# 5. Описать основные компоненты, из которых состоит нейросетевой блок генерации дикторских моделей (фреймовый уровень, слой статистического пулинга, сегментный уровень, выходной слой).
# 
# 6. Как устроены нейросетевые архитектуры на основе ResNet-блоков?
# 
# 7. Что такое полносвязная нейронная сеть прямого распространения?
# 
# 8. Как устроена стоимостная функция для обучения нейросетевого блока генерации дикторских моделей?
# 
# 9. Что такое аугментация данных?
# 
# 10. Что такое дикторский эмбеддинг и на каком уровне блока построения дикторских моделей он генерируется?

# **6. Список литературы**
# 
# 1. Bai Z., Zhang X.-L., Chen J. Speaker recognition based on deep learning: an overview // 	arXiv:2012.00931 [eess.AS] ([ссылка](https://arxiv.org/pdf/2012.00931.pdf)).
# 
# 2. Hansen J.H.L., Hasan T. Speaker recognition by machines and humans: a tutorial review // IEEE Signal Processing Magazine, 2015. V. 32. № 6. P. 74–99 ([ссылка](https://www.researchgate.net/publication/282940395_Speaker_Recognition_by_Machines_and_Humans_A_tutorial_review)).
