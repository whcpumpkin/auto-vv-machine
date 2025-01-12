# Auto-vv-Machine

张维为教授总能一针见血地指出社会的问题，学养深厚，深受大家喜爱。我整理了张教授的诸多语录，可以使用关键词或者BGE-M3模型检索，帮助大家日常学习。


![demo.png](demo.png)

## 资源下载
下载地址：

百度网盘：https://pan.baidu.com/s/1oR9d-Yx_j7U013C6eaKCsg?pwd=wkgj 

下载之后解压zip包。

## 目录结构
```
├── README.md
├── search.py
├── requirements.txt
├── bge-m3
├── name.pth
├── cached_file_fp16.pth
├── demo.png
└── filtered_frames
     ├── xx.jpg
     ├── yy.jpg
     ├── zz.jpg
```


## 环境安装
### 普通模式
```

conda create -n auto-vv-machine python=3.10
conda activate auto-vv-machine

(cuda版本和cpu版本请根据硬件选择其中一个)
cuda版本：
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
cpu版本：
pip install torch==2.3.0

pip install -r requirements.txt
```

### AI 模式
```
在普通模式基础上安装：
pip install -U FlagEmbedding
git clone https://huggingface.co/BAAI/bge-m3 // 这一步不进行会自动联网下载，注意你的网络情况
```

注意：AI模式可以不安装，仅使用关键词搜索。如果启用AI模式+CUDA，建议空闲显存至少为6G；如果启用AI模式+CPU，建议空闲内存至少为8G

## 使用

```
python search.py
```

运行之后会有一个弹窗，选择截图文件夹``filtered_frames``，然后点击一下``Search``按钮，激活一下输入框，才能开始检索（这个“激活”BUG待解决）

可以在最上方设置设备为cpu或者cuda:0，选择之后点一下``Set device``按钮。

每次运行后，第一次用AI检索会有点慢，是因为模型还在导入。

