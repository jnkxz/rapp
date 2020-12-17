# -*- coding:utf-8 -*-
# Author:weirong, zwj, Jnk_xz,
# pylint: disable=no-member
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage as nd


def circle(img, X, Y, radius):
    '''
    根据给出的星的位置进行画圈
    获得所画的圈中的像素的值
    '''
    h, w = img.shape
    y = np.arange(int(np.around(Y - radius)),
                  int(np.around(Y + radius)))
    y = np.intersect1d(np.arange(h), y)
    x = np.arange(int(np.around(X - radius)),
                  int(np.around(X + radius)))
    x = np.intersect1d(np.arange(w), x)
    x, y = np.meshgrid(x, y)
    mask = (((Y - y)**2 + (X - x)**2)**0.5 <= radius)
    values = img[y[mask], x[mask]]
    return values


def annular(img, X, Y, inner, outer):
    '''
    根据给出的星的位置进行画环
    获得所画的环中的像素的值
    '''
    h, w = img.shape
    y = np.arange(int(np.around(Y - outer)),
                  int(np.around(Y + outer)))
    y = np.intersect1d(np.arange(h), y)
    x = np.arange(int(np.around(X - outer)),
                  int(np.around(X + outer)))
    x = np.intersect1d(np.arange(w), x)
    x, y = np.meshgrid(x, y)
    radius = ((Y - y)**2 + (X - x)**2)**0.5
    mask = (inner <= radius) * (radius <= outer)
    values = img[y[mask], x[mask]]
    return values


def ap(img: np.ndarray, x: float, y: float, r: float,
       gain: float, aperture: float, inner: float, outer: float):
    h, w = img.shape
    yy = np.arange(int(np.around(y - r*outer)),
                   int(np.around(y + r*outer)))
    yy = np.intersect1d(np.arange(h), yy)
    xx = np.arange(int(np.around(x - r*outer)),
                   int(np.around(x + r*outer)))
    xx = np.intersect1d(np.arange(w), xx)
    xx, yy = np.meshgrid(xx, yy)
    distence = np.sqrt((xx - x)**2 + (yy - y)**2)

    mrk = (distence < r*aperture)
    adu = img[yy[mrk], xx[mrk]]

    mrk = (r*inner <= distence) & (distence <= r*outer)
    sky = img[yy[mrk], xx[mrk]]

    med, std = np.median(sky), np.std(sky)
    flx = np.sum(adu - med)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flx)
        err = 2.5 / np.log(10) / flx * np.sqrt(
            + flx / gain
            + len(adu) * np.square(std)
            + np.square(len(adu)) * np.square(std) / len(sky))
        snr = flx / (np.sqrt(len(adu)) * std)
    return flx, mag, err, snr


def find_star(hdu: fits.HDUList, bias: fits.HDUList = None, flat: fits.HDUList = None, n: float = 3.,
              aperture: float = 1.2, inner: float = 2.4, outer: float = 3.6):
    '''
    从星图中找星
    para:
        hdu:HDUList         输入的星图
        bias:HDUList        本底文件默认None
        flat:HDUList        平场文件默认None
        n:float             sigma倍数默认3
        aperture:float      孔径对半径倍数默认1.2
        inner:float         背景内径对半径倍数默认2.4
        outer:float         背景外径对半径倍数默认3.6
    return:
        result:ndarray      输出结果
            idx: int        星表编号(当前为-1)
            x: float        图片中的x坐标
            y: float        图片中的y坐标
            r: float        图片中的星象半径
            ra: float       星坐标对应的ra
            dec: float      星坐标对应的dec
            ra_m: float     匹配到的星的ra(当前为NaN)
            dec_m: float    匹配到的星的dec(当前为NaN)
            m_err: float    与匹配星偏差角度(当前为NaN)
            flux: float     adu总流量
            mag: float      仪器星等
            mag_m: float    星表星等(当前为NaN)
            err: float      仪器星等误差
            snr: float      星在图片中的信噪比
    '''
    du = hdu[1].data.astype(float)
    if bias is not None:
        bu = bias[1].data.astype(float)
    else:
        bu = 0

    if flat is not None:
        fu = flat[1].data.astype(float) - bu
        fu = fu / np.median(fu)
    else:
        fu = 1
    du = (du - bu) / fu

    hu = hdu[1].header
    gain = float(hu['GAIN'])

    wcs = WCS(hu)

    m, s = np.median(du), np.std(du)
    mrk = du > m + n * s

    lbl, num = nd.measurements.label(mrk)

    idx = np.arange(num) + 1
    def rfunc(x): return np.sqrt(len(x)/np.pi)
    R = nd.labeled_comprehension(input=lbl,
                                 labels=lbl,
                                 index=idx,
                                 func=rfunc,
                                 out_dtype=float,
                                 default=0)
    i = R >= 1
    idx = idx[i]
    R = R[i]

    dtype = np.dtype([('idx', np.int),
                      ('x', np.float),
                      ('y', np.float),
                      ('r', np.float),
                      ('ra', np.float),
                      ('dec', np.float),
                      ('ra_m', np.float),
                      ('dec_m', np.float),
                      ('m_err', np.float),
                      ('flux', np.float),
                      ('mag', np.float),
                      ('mag_m', np.float),
                      ('err', np.float),
                      ('snr', np.float), ])
    result = np.empty(len(R), dtype)
    # 计算各连通域重心
    C = nd.measurements.center_of_mass(input=du,
                                       labels=lbl,
                                       index=idx)
    C = np.array(C)
    X, Y = C[:, 1], C[:, 0]

    for i, item in enumerate(zip(X, Y, R)):
        x, y, r = item
        flx, mag, err, snr = ap(du, x, y, r, gain, aperture, inner, outer)
        coord = wcs.pixel_to_world(x, y)
        result['idx'][i] = -1
        result['x'][i] = x
        result['y'][i] = y
        result['r'][i] = r
        result['ra'][i] = coord.ra.degree
        result['dec'][i] = coord.dec.degree
        result['ra_m'][i] = np.nan
        result['dec_m'][i] = np.nan
        result['m_err'][i] = np.nan
        result['flux'][i] = flx
        result['mag'][i] = mag
        result['mag_m'][i] = np.nan
        result['err'][i] = err
        result['snr'][i] = snr
    return result


def match(result: np.ndarray, catalog: fits.HDUList):
    '''
    将结果与星表匹配
    para:
        result:ndarray      输入的星图
        cat:HDUList         星表文件
    return:
        result:ndarray      输出结果
            idx: int        星表编号
            x: float        图片中的x坐标
            y: float        图片中的y坐标
            r: float        图片中的星象半径
            ra: float       星坐标对应的ra
            dec: float      星坐标对应的dec
            ra_m: float     匹配到的星的ra
            dec_m: float    匹配到的星的dec
            m_err: float    与匹配星偏差角度
            flux: float     adu总流量
            mag: float      仪器星等
            mag_m: float    星表星等
            err: float      仪器星等误差
    '''
    cat = catalog[1].data
    stars = SkyCoord(ra=cat['ra']*u.deg, dec=cat['dec']*u.deg)
    coords = SkyCoord(ra=result['ra']*u.deg, dec=result['dec']*u.deg)
    idx, err, _ = coords.match_to_catalog_sky(stars)
    matchs = cat[idx]
    result['idx'] = idx
    result['m_err'] = err
    result['ra_m'] = matchs['ra']
    result['dec_m'] = matchs['dec']
    result['mag_m'] = matchs['mag_i']
    return result
