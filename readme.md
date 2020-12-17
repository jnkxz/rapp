# 连通域找星程序

## 简介

利用连通域找星, 再根据连通域的半径进行孔径测光得到结果
再用结果与星表匹配, 得到对应星表结果

## 安装

```bash
python setup.py install
```

## 使用

```python
from rapp.api import find_star, match

fits_res = find_star(hdu)
cata_res = match(fits_res, cat)
```