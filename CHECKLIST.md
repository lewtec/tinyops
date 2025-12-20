# CHECKLIST

## linalg

- [x] `np.dot` → `tinyops.linalg.dot`
- [x] `np.matmul` → `tinyops.linalg.matmul`
- [x] `np.vdot` → `tinyops.linalg.vdot`
- [x] `np.inner` → `tinyops.linalg.inner`
- [x] `np.outer` → `tinyops.linalg.outer`
- [x] `np.tensordot` → `tinyops.linalg.tensordot`
- [x] `np.einsum` → `tinyops.linalg.einsum`
- [x] `np.linalg.inv` → `tinyops.linalg.inv`
- [x] `np.linalg.pinv` → `tinyops.linalg.pinv`
- [x] `np.linalg.solve` → `tinyops.linalg.solve`
- [x] `np.linalg.lstsq` → `tinyops.linalg.lstsq`
- [ ] `np.linalg.det` → `tinyops.linalg.det`
- [x] `np.linalg.norm` → `tinyops.linalg.norm`
- [x] `np.linalg.cond` → `tinyops.linalg.cond`
- [ ] `np.linalg.matrix_rank` → `tinyops.linalg.matrix_rank`
- [x] `np.trace` → `tinyops.linalg.trace`
- [ ] `np.linalg.eig` → `tinyops.linalg.eig`
- [ ] `np.linalg.eigh` → `tinyops.linalg.eigh`
- [ ] `np.linalg.eigvals` → `tinyops.linalg.eigvals`
- [ ] `np.linalg.svd` → `tinyops.linalg.svd`
- [ ] `np.linalg.qr` → `tinyops.linalg.qr`
- [ ] `np.linalg.cholesky` → `tinyops.linalg.cholesky`
- [x] `np.linalg.matrix_power` → `tinyops.linalg.matrix_power`
- [x] `np.kron` → `tinyops.linalg.kron`
- [x] `np.diagonal` → `tinyops.linalg.diagonal`

## stats

- [x] `np.mean` → `tinyops.stats.mean`
- [x] `np.median` → `tinyops.stats.median`
- [x] `np.std` → `tinyops.stats.std`
- [x] `np.var` → `tinyops.stats.var`
- [x] `np.average` → `tinyops.stats.average`
- [x] `np.percentile` → `tinyops.stats.percentile`
- [x] `np.quantile` → `tinyops.stats.quantile`
- [ ] `np.ptp` → `tinyops.stats.ptp`
- [ ] `np.histogram` → `tinyops.stats.hist`
- [ ] `np.histogram2d` → `tinyops.stats.hist2d`
- [ ] `np.histogramdd` → `tinyops.stats.histdd`
- [ ] `np.bincount` → `tinyops.stats.bincount`
- [ ] `np.digitize` → `tinyops.stats.digitize`
- [ ] `np.corrcoef` → `tinyops.stats.corrcoef`
- [ ] `np.correlate` → `tinyops.stats.correlate`
- [ ] `np.cov` → `tinyops.stats.cov`

## image

- [ ] `cv2.resize` → `tinyops.image.resize`
- [ ] `cv2.rotate` → `tinyops.image.rotate`
- [ ] `cv2.flip` → `tinyops.image.flip`
- [ ] `cv2.warpAffine` → `tinyops.image.warp_affine`
- [ ] `cv2.warpPerspective` → `tinyops.image.warp_perspective`
- [ ] `cv2.blur` → `tinyops.image.blur`
- [ ] `cv2.boxFilter` → `tinyops.image.box_filter`
- [ ] `cv2.GaussianBlur` → `tinyops.image.gaussian_blur`
- [ ] `cv2.medianBlur` → `tinyops.image.median_blur`
- [ ] `cv2.bilateralFilter` → `tinyops.image.bilateral_filter`
- [ ] `cv2.filter2D` → `tinyops.image.filter2d`
- [ ] `cv2.Sobel` → `tinyops.image.sobel`
- [ ] `cv2.Scharr` → `tinyops.image.scharr`
- [ ] `cv2.Laplacian` → `tinyops.image.laplacian`
- [ ] `cv2.Canny` → `tinyops.image.canny`
- [ ] `cv2.dilate` → `tinyops.image.dilate`
- [ ] `cv2.erode` → `tinyops.image.erode`
- [ ] `cv2.morphologyEx` → `tinyops.image.morphology`
- [ ] `cv2.threshold` → `tinyops.image.threshold`
- [ ] `cv2.adaptiveThreshold` → `tinyops.image.adaptive_threshold`
- [ ] `cv2.cvtColor` → `tinyops.image.cvt_color`
- [ ] `cv2.equalizeHist` → `tinyops.image.equalize_hist`
- [ ] `cv2.normalize` → `tinyops.image.normalize`
- [ ] `torchvision.transforms.CenterCrop` → `tinyops.image.center_crop`
- [ ] `torchvision.transforms.Pad` → `tinyops.image.pad`
- [ ] `torchvision.transforms.ColorJitter` → `tinyops.image.color_jitter`

## audio

- [ ] `torchaudio.transforms.Spectrogram` → `tinyops.audio.spectrogram`
- [ ] `torchaudio.transforms.MelSpectrogram` → `tinyops.audio.mel_spectrogram`
- [ ] `torchaudio.transforms.MFCC` → `tinyops.audio.mfcc`
- [ ] `torchaudio.transforms.MelScale` → `tinyops.audio.mel_scale`
- [ ] `torchaudio.transforms.AmplitudeToDB` → `tinyops.audio.amplitude_to_db`
- [ ] `torchaudio.transforms.GriffinLim` → `tinyops.audio.griffin_lim`
- [ ] `torchaudio.transforms.Resample` → `tinyops.audio.resample`
- [ ] `torchaudio.transforms.MuLawEncoding` → `tinyops.audio.mu_law_encode`
- [ ] `torchaudio.transforms.MuLawDecoding` → `tinyops.audio.mu_law_decode`
- [ ] `torchaudio.transforms.TimeStretch` → `tinyops.audio.time_stretch`
- [ ] `torchaudio.transforms.FrequencyMasking` → `tinyops.audio.freq_mask`
- [ ] `torchaudio.transforms.TimeMasking` → `tinyops.audio.time_mask`
- [ ] `torchaudio.transforms.Fade` → `tinyops.audio.fade`

## signal

- [ ] `np.fft.fft` → `tinyops.signal.fft`
- [ ] `np.fft.ifft` → `tinyops.signal.ifft`
- [ ] `np.fft.fft2` → `tinyops.signal.fft2`
- [ ] `np.fft.ifft2` → `tinyops.signal.ifft2`
- [ ] `np.fft.rfft` → `tinyops.signal.rfft`
- [ ] `np.fft.irfft` → `tinyops.signal.irfft`
- [ ] `np.fft.fftfreq` → `tinyops.signal.fftfreq`
- [ ] `np.convolve` → `tinyops.signal.convolve`
- [ ] `np.correlate` → `tinyops.signal.correlate`
- [ ] `np.hanning` → `tinyops.signal.hanning`
- [ ] `np.hamming` → `tinyops.signal.hamming`
- [ ] `np.blackman` → `tinyops.signal.blackman`
- [ ] `np.kaiser` → `tinyops.signal.kaiser`

## io

- [ ] wav decode → `tinyops.io.decode_wav`
- [ ] wav encode → `tinyops.io.encode_wav`
- [ ] bmp decode → `tinyops.io.decode_bmp`
- [ ] bmp encode → `tinyops.io.encode_bmp`
