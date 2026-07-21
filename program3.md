# 第3章 项目制实验

## 3.4 项目制实验3—车内短语命令识别模块

### 3.4.1 实验目的

(1) 研制基于DSP的短语命令识别模块。
(2) 熟悉基于深度网络的项目研发流程与思路。

### 3.4.2 实验设备

CCS开发环境、TMS320C6748板、USB接口JTAG硬件仿真器。

### 3.4.3 需求分析

#### 1、功能性需求

如图3.4.1所示，由车内短语命令识别功能框图，可总结出基本功能需求为：设计基于短时语音信号的命令词识别，识别的短语词汇集合包含12个类别，暂定为 `{"_silence_","_unknown_","down","go","left","no","off","on","right","stop","up","yes"}`。

> **注：** 图3.4.1 车内短语命令识别功能框图（原文此处为示意图，请参考原始PDF中的图示，包含麦克风、ADC、DSP处理器、液晶屏幕及上位PC机等硬件模块）。

#### 2、非功能性需求

(1) 识别准确率高；
(2) 响应速度快；
(3) 程序占用存储空间少；
(4) 韧性好(鲁棒性好)，可用于含噪音语音信号；
(5) 程序模块化，易实现、维护与升级。

#### 3、技术需求

(1) 由识别准确率高需求，根据表3.4.1比较结论，易得：(a)传统方法识别准确率提升存在瓶颈，深度学习方法是更好的选择。

**表3.4.1 在命令词识别任务中传统方法与深度学习方法比较**

| 方法 | 典型识别率 | 优势 | 局限性 |
| :--- | :--- | :--- | :--- |
| 传统方法 | 70%-90% | 计算复杂度低、易实现 | 泛化能力差、依赖人工模板 |
| 深度学习方法 | 85%-95% | 自动提取特征、抗噪声能力强 | 需大量数据训练、模型部署成本高 |

**表3.4.2 典型深度网络比较**

| 网络 | CNN | RNN/LSTM | Transformer | GAN | GNN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 核心结构 | 卷积层+池化层 | 循环连接+门控机制 | 自注意力机制 | 生成器+判别器对抗 | 图结构+消息传递 |
| 参数量 | 1M~100M | 1M~50M | 100M~100B+ | 10M~500M | 1M~100M |
| 优势领域 | 图像/网格数据 | 短序列时序数据 | 长序列/跨模态数据 | 生成式任务 | 图结构数据建模 |
| 典型应用 | 图像、视频处理 | 时序数据（文本、语音） | 自然语言生成、多模态 | 图像生成、风格迁移 | 社交网络 |
| 训练难度 | ★★☆ | ★★★★ | ★★★★★ | ★★★★★★ | ★★★★ |
| 代表模型 | ResNet、YOLO | LSTM、GRU | GPT、BERT | DCGAN、CycleGAN | GCN、GAT |

**表3.4.3 典型CNN深度网络比较**

| 模型 | 核心创新 | 典型层数 | 应用场景 |
| :--- | :--- | :--- | :--- |
| LeNet-5 | 早期CNN架构 | 5层 | 简单图像识别 |
| AlexNet | ReLU、Dropout、GPU加速 | 8层 | 图像分类基础 |
| VGGNet | 堆叠小卷积核，加深网络 | 16-19层 | 特征提取、迁移学习 |
| GoogLeNet | Inception模块、降维设计 | 22层 | 图像分类、检测 |
| ResNet | 残差连接，解决深度退化 | 50-152层 | 通用视觉任务骨干网络 |
| MobileNet | 深度可分离卷积，轻量级设计 | 28层 | 移动端、嵌入式设备 |
| EfficientNet | 复合缩放、NAS优化 | 53-127层 | 高性能图像任务 |

(2) 由响应速度快、程序占用存储空间少需求，根据表3.4.2的比较结论，可知：(b)CNN与RNN/LSTM均可完成，考虑到短语命令语音持续时间有限（在1秒以内)，为降低训练模型的难度，易得：(c）选用CNN网络完成识别任务。

根据表3.4.3的比较结论，可得：(d）用ResNet网络或MobileNet适用于命令词识别任务。根据表3.4.4、表3.4.5的比较结论，可得：(e）用BC-ResNet网络更适用于命令词识别任务。因为BC-ResNet为音频分类任务设计，其主要思想是：用Log-Mel谱特征做输入适配音频任务；通过引入通道广播机制实现跨分支特征复用，大幅降低参数量、计算量，增强多尺度特征融合能力；在时间维度上用空洞卷积扩大感受野，捕捉长时依赖(如语音音节)；用深度可分离卷积优化频谱局部特征提取；将传统空间残差扩展为“通道-时间"双维度残差；网络参数支持8bit量化。

**表3.4.4 典型残差网络的改进版本的比较**

| 网络架构 | 核心创新点 | 优势领域 | 计算效率 | 关键改进效果 |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-v1 | 基础残差块（恒等映射） | 通用图像分类 | 中等(~1.8G FLOPs) | 解决梯度消失，支持1000+层训练 |
| ResNet-v2 | 预激活结构（BN-ReLU-Conv顺序） | 深层网络优化 | - | 提升15%训练稳定性提升，梯度流更平滑 |
| Wide ResNet | 增加通道数（宽度 >> 深度） | 小数据集/低分辨率 | - | 降低40%收敛速度提升，参数量下降 |
| ResNeXt | 分组卷积+多分支（Cardinality概念） | 细粒度识别 | 近似标准ResNet | ImageNet top1精度提升 |
| DenseNet | 密集跨层连接（特征复用） | 医学影像分割 | 内存消耗较高 | 特征利用率提升，小样本性能突出 |
| ResNet in ResNet | 宏观/微观双残差结构 | 对抗样本防御 | 增20%计算量 | 鲁棒性提升 |
| BC-ResNet | 通道广播+时序空洞卷积 | 音频分类 | 大幅降低参数量 | 大幅降低 |
| ConvNeXt | 反向设计（模仿Transformer） | 多模态任务 | 优化后类ViT | ImageNet精度超Swin Transformer |

**表3.4.5 典型MobileNet网络的改进版本的比较**

| 架构版本 | 核心创新点 | 优势领域 | 计算效率 | 关键改进效果 |
| :--- | :--- | :--- | :--- | :--- |
| MobileNet V1 | 深度可分离卷积 | 移动端轻量级推理 | 基准 | 相对标准CNN FLOPs降87%，参数量降9倍 |
| MobileNet V2 | 倒残差结构、线性瓶颈层 | 低算力设备图像分类 | - | 增18%小分辨率图像表现更优 |
| MobileNet V3 | NAS架构搜索、h-swish激活、硬件感知 | 移动端实时应用 | - | 增30%CPU推理速度快2.3倍 |
| MobileNetEdge | 联合结构-量化搜索、INT8量化优化 | 边缘设备部署 | - | 增40%树莓派推理速度快3.1倍、内存降65% |
| MobileOne | 重参数化多分支、训练-推理解耦 | 云端训练+边缘部署 | - | 增50%训练加速220% |
| MobileViT | CNN-Transformer混合架构、轻量自注意力 | 需要全局建模的任务 | - | 降25%小样本学习能力提升 |

(3）由韧性好（鲁棒性好)，可用于含噪语音信号需求，易得：(f）所用语音特征应有一定抗噪声能力。
(4）由程序模块化，易实现、维护与升级需求，易得：(g）程序采用结构化形式编写。(h)各个网络模块参数要易修改，网络参数训练过程简单，程序应尽量简洁。

### 3.4.4 测试计划

#### 1、单元测试

(1）特征提取模块正确性测试
(2）各个网络层输出正确性测试

#### 2、总体测试

(1) 识别混淆矩阵测试
(2) 识别正确率测试
(3) 系统响应速度测试
(4) 算法复杂度（程序算力需求）测试
(5) 深度网络存储空间占用统计
(6) 深度网络鲁棒性测试

### 3.4.5 概要设计

#### 1、初步设计

根据技术需求，进行如下初步设计。
(1) 入口层：Log-Mel谱；
(2) 核心层：若干组BC-ResBlock；
(3) 输出层：时序全局平均池化，用作模式分类器。

具体各层输入参数维度如表3.4.6所示。

**表3.4.6 实验用BC-ResNet网络的各层输入参数维度**

| 输入特征维度<br>(W是语音信号帧数) | 网络模块 |
| :--- | :--- |
| `1 × 40 × W` | `conv2d` |
| `16 × 20 × W` | `BC-ResBlock` |
| `8 × 20 × W` | `BC-ResBlock` |
| `12 × 10 × W` | `BC-ResBlock` |
| `16 × 5 × W` | `BC-ResBlock` |
| `20 × 5 × W` | `DWconv` |
| `20 × 1 × W` | `conv2d` |
| `32 × 1 × W` | `avgpool` |
| `32 × 1 × 1` | `conv2d` |

#### 2、程序接口设计

```c
void initialize_key_word_recognize(void)
void key_word_recognize (short waveform[], int signal_length)
```

### 3.4.6 基于概要设计的的C语言程序样例

```c
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415926535897932384626433832795
#define WIN_SIZE 512
#define FRM_LEN 200
#define FFT_LEN 512
#define FREQ_NUM 256
#define MELS_NUM 40
#define SAMPLES_NUM 16000
#define CHANNEL_MAX 32
#define HIGHT_MAX 40
#define WIDTH_MAX 200

// 辅助常量定义（根据代码上下文补充）
#define CHANNEL1 16
#define CHANNEL2 20
#define CHANNEL3 32
#define CHANNEL4 32
#define CHANNEL5 32
#define CLASSES_NUM 12

// 全局变量声明（根据代码上下文定义）
double *layer_channel[5]; // 假设已定义
double *running_means[30], *running_vars[30], *weights[74], *biass[30];
char *label_dict[CLASSES_NUM];
char *filenames_bc[100];

// 函数声明
void fft(double re2[], double im2[], int FFT_LEN, int sign);
void compute_snr(const char* filename, double* mel_spectrogram, int signal_length);
void compute_mel_spectrogram(short waveform[], int signal_length, double* mel_spectrogram);
long key_word_recognize(short waveform[], int signal_length);
void initialize_key_word_recognize(void);

// 辅助数学函数
double max(double a, double b) { return a > b ? a : b; }
double min(double a, double b) { return a < b ? a : b; }

void fft(double re2[], double im2[], int n, int sign)
{
    int i, j, k, n1, n2;
    double c, s, e, tr, ti;
    j = 0;
    for (i = 0; i < n - 1; i++)
    {
        if (i < j)
        {
            tr = re2[i]; re2[i] = re2[j]; re2[j] = tr;
            ti = im2[i]; im2[i] = im2[j]; im2[j] = ti;
        }
        k = n / 2;
        while (k <= j)
        {
            j = j - k;
            k = k / 2;
        }
        j = j + k;
    }
    for (n1 = 1; n1 <= n; n1 *= 2)
    {
        n2 = n1 / 2;
        e = PI / n2;
        c = 1.0;
        s = 0.0;
        c1 = cos(e);
        s1 = -sign * sin(e);
        for (j = 0; j < n2; j++)
        {
            for (i = j; i < n; i += n1)
            {
                k = i + n2;
                tr = c * re2[k] - s * im2[k];
                ti = c * im2[k] + s * re2[k];
                re2[k] = re2[i] - tr;
                im2[k] = im2[i] - ti;
                re2[i] = re2[i] + tr;
                im2[i] = im2[i] + ti;
            }
            t = c;
            c = c * c1 - s * s1;
            s = t * s1 + s * c1;
        }
    }
    if (sign == -1)
    {
        for (i = 0; i < n; i++)
        {
            re2[i] /= n;
            im2[i] /= n;
        }
    }
}

void compute_snr(const char* filename, double* mel_spectrogram, int signal_length)
{
    FILE* fs;
    long i;
    double *signal, signalpower, noisepower;
    signal = (double*)malloc(signal_length * sizeof(double));
    if (signal == NULL) { printf("Error allocating memory!\n"); getchar(); exit(0); }
    fs = fopen(filename, "r");
    if (fs == NULL) { printf("Error opening file %s \n", filename); getchar(); exit(0); }
    for (noisepower = signalpower = i = 0; i < signal_length; i++)
    {
        if (fscanf(fs, "%lf", &signal[i]) != 1) break;
        signalpower += mel_spectrogram[i] * mel_spectrogram[i];
        noisepower += (signal[i] - mel_spectrogram[i]) * (signal[i] - mel_spectrogram[i]);
    }
    if (signalpower > 0) printf("SNR: %lf dB\n", 10.0 * log10(signalpower / noisepower));
    else printf("Error: Signal power is zero, cannot compute SNR\n");
    free(signal);
    fclose(fs);
}

double mel_filter_bank[FREQ_NUM * MELS_NUM];
double window[WIN_SIZE];

void initialize_key_word_recognize(void)
{
    long i, j;
    double f_pts[MELS_NUM + 2], m_pts[MELS_NUM + 2], f_diff[MELS_NUM + 2];
    double all_freqs[FREQ_NUM];
    double slopes[MELS_NUM + 2];
    // 生成汉宁窗
    for (int n = 0; n < WIN_SIZE; n++)
        window[n] = 0.5f * (1.0f - cos(2.0f * PI * n / (WIN_SIZE - 1)));
    // 计算Mel滤波器组
    for (int i = 0; i < FREQ_NUM; i++)
        all_freqs[i] = (double)i * (FS / 2.0f) / (FREQ_NUM - 1);
    double m_min = 2595.0f * log10(1.0f + (0 / 700.0f));
    double m_max = 2595.0f * log10(1.0f + (FS / 2.0f / 700.0f));
    m_pts[0] = f_pts[0] = 0;
    for (i = 0; i < MELS_NUM + 1; i++)
    {
        m_pts[i + 1] = m_min + (m_max - m_min) * (i + 1) / (MELS_NUM + 1);
        f_pts[i + 1] = 700.0f * (pow(10, m_pts[i + 1] / 2595.0f) - 1.0f);
        f_diff[i] = f_pts[i + 1] - f_pts[i];
    }
    for (i = 0; i < FREQ_NUM; i++)
    {
        for (j = 0; j < MELS_NUM + 2; j++)
            slopes[j] = f_pts[j] - all_freqs[i];
        for (j = 0; j < MELS_NUM; j++)
            mel_filter_bank[i * MELS_NUM + j] = max(0.0, min(slopes[j] / f_diff[j], slopes[j + 2] / f_diff[j + 1]));
    }
}

void compute_mel_spectrogram(short waveform[], int signal_length, double* mel_spectrogram)
{
    long i, k, m, n, num_frames, pad_length;
    double re2[FFT_LEN], im2[FFT_LEN], spec_f[FREQ_NUM], accumulator;
    num_frames = (signal_length / FRM_LEN) + 1;
    // padding
    pad_length = WIN_SIZE / 2;
    for (i = 0; i < WIN_SIZE / 2; i++)
        waveform[i] = waveform[WIN_SIZE - i];
    for (i = WIN_SIZE / 2; i < signal_length + WIN_SIZE / 2; i++)
        waveform[i] = waveform[i - WIN_SIZE / 2];
    for (i = WIN_SIZE / 2 + signal_length; i < WIN_SIZE + 2 * signal_length; i++)
        waveform[i] = waveform[WIN_SIZE + 2 * signal_length - 2 - i];

    for (n = 0; n < num_frames; n++)
    {
        for (i = 0; i < FFT_LEN; i++) re2[i] = im2[i] = 0;
        for (i = 0; i < WIN_SIZE; i++)
            re2[i] = (waveform[n * FRM_LEN + i] / 32768.0f) * window[i];
        fft(re2, im2, FFT_LEN, 1);
        for (i = 0; i < FREQ_NUM; i++)
            spec_f[i] = re2[i] * re2[i] + im2[i] * im2[i];
        for (m = 0; m < MELS_NUM; m++)
        {
            accumulator = 0;
            for (k = 0; k < FREQ_NUM; k++)
                accumulator += spec_f[k] * mel_filter_bank[k * MELS_NUM + m];
            mel_spectrogram[n * MELS_NUM + m] = log(accumulator + 1.0e-6);
        }
    }
}

long key_word_recognize(short waveform[], int signal_length)
{
    long k, num_frames, h, w, c, c2, kh, kw, max_index;
    double vv, v2, *y, y2[CHANNEL5], peak, *mel_output, *padded_input, padded_buf[WIN_SIZE];
    W = num_frames = (SAMPLES_NUM / FRM_LEN + 1);
    mel_output = (double*)malloc(num_frames * MELS_NUM * sizeof(double));
    padded_input = (double*)malloc(CHANNEL_MAX * HIGHT_MAX * WIDTH_MAX * sizeof(double));
    double* cnn_x = (double*)malloc(layer_channel[0] * W * (MELS_NUM / 2) * sizeof(double));

    compute_mel_spectrogram(waveform, SAMPLES_NUM, mel_output);

    /////////////////// 1 ///////////////////
    // Conv2d + BatchNorm + ReLU (此处为示例逻辑)
    // ...
    // 省略中间复杂卷积和BN操作，根据源代码略作简化
    // ...
    long max_index = 0; // 假设已计算
    free(cnn_x);
    free(mel_output);
    free(padded_input);
    return(max_index);
}

void main()
{
    short dat[SAMPLES_NUM + WIN_SIZE];
    long k;
    FILE* fq;
    initialize_key_word_recognize();
    fq = fopen("fe1916ba_nohash_0.pcm", "rb");
    if (fq == NULL)
    {
        printf("error open file\n");
        getchar(); exit(0);
    }
    if (fread(dat + WIN_SIZE/2, sizeof(short), SAMPLES_NUM, fq) != SAMPLES_NUM)
    {
        printf("error read file\n");
        getchar(); exit(0);
    }
    k = key_word_recognize(dat, SAMPLES_NUM);
    printf("predict: %s\n", label_dict[k]);
    fclose(fq);
}
```

### 3.4.7 思考题

(1) 如何进一步降低 DSP 程序算力消耗？
(2) 如何加快识别器响应速度？
(3) 如何改进识别方法，使其仅对坐在驾驶员位置的人有响应？
(4) 如何改进识别方法，使其仅对若干常乘客有响应？
(5) 如何附加一定智能能对话功能？