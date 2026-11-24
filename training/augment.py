"""Augmentation for 256x512 dark-on-light signature images (train split only).

Input/output: uint8 grayscale numpy array, output always (256, 512), white fill = 255.
Order: rotation -> scale/shift -> thickness -> ink lightness -> junk -> brightness/contrast -> blur -> noise/JPEG
Each transform fires independently, so most images get 1-3 changes and some stay untouched.
Settings are drawn per image. The class never sees the label, so genuine and forged
images get identical chances (the model cannot learn "distorted = forged").

Group 1 (default): rotation, scale+shift, thickness, lighter ink, blur, noise/JPEG
Group 2 (group2=True, add only after Group 1 works): junk pasting, brightness/contrast

Not included on purpose: flips, elastic/heavy warps, cutout/erasing, colour changes.
"""
import os
import cv2
import numpy as np

H, W = 256, 512


class SignatureAugment:
    def __init__(self, group2=False, seed=None,
                 p_rot=0.40, p_rot_big=0.10, rot_small=8.0, rot_big=15.0,
                 p_scaleshift=0.40, scale=0.10, shift=0.05,
                 p_thick=0.25, p_thin=0.15,
                 p_light=0.20, light_max=0.40,
                 p_blur=0.25, blur_sigma=(0.5, 1.0),
                 p_noise_jpeg=0.25, noise_sigma=(2.0, 5.0), jpeg_q=(60, 90),
                 p_junk=0.12,
                 p_bc=0.15):
        self.group2 = group2
        self.seed = seed
        self.epoch = 0
        self._rng = None
        self.p_rot, self.p_rot_big, self.rot_small, self.rot_big = p_rot, p_rot_big, rot_small, rot_big
        self.p_scaleshift, self.scale, self.shift = p_scaleshift, scale, shift
        self.p_thick, self.p_thin = p_thick, p_thin
        self.p_light, self.light_max = p_light, light_max
        self.p_blur, self.blur_sigma = p_blur, blur_sigma
        self.p_noise_jpeg, self.noise_sigma, self.jpeg_q = p_noise_jpeg, noise_sigma, jpeg_q
        self.p_junk, self.p_bc = p_junk, p_bc

    @property
    def rng(self):
        # created lazily so each DataLoader worker process gets its own random stream
        if self._rng is None:
            self._rng = np.random.default_rng(
                None if self.seed is None else [self.seed, os.getpid()])
        return self._rng

    def set_epoch(self, epoch: int) -> None:
        """Choose the stream used for ``epoch``.

        With ``seed`` set and a per-sample ``index`` passed to :meth:`__call__`, every
        augmentation becomes a pure function of ``(seed, epoch, index)``. The same
        sample then gets the same augmentation however the batches are ordered, how
        many DataLoader workers run, and in which process. Without this, seeding on
        ``os.getpid()`` alone replays one identical pattern every epoch.
        """
        self.epoch = int(epoch)

    def _u(self, a, b):
        return float(self.rng.uniform(a, b))

    # 1 + 2: rotation and scale/shift, combined into one warp (one interpolation, same result)
    def _geometry(self, img, do_rot, do_ss, ops):
        ang, sc, tx, ty = 0.0, 1.0, 0.0, 0.0
        if do_rot:
            lim = self.rot_big if self.rng.random() < self.p_rot_big else self.rot_small
            ang = self._u(-lim, lim)
            ops.append("rotation")
        if do_ss:
            sc = 1.0 + self._u(-self.scale, self.scale)
            tx, ty = self._u(-self.shift, self.shift) * W, self._u(-self.shift, self.shift) * H
            ops.append("scale_shift")
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
        M[:, 2] += (tx, ty)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 3: stroke thickness by 1 px (ink is dark: erode = thicker, dilate = thinner)
    def _thickness(self, img, thick):
        k = np.ones((2, 2), np.uint8)
        return cv2.erode(img, k) if thick else cv2.dilate(img, k)

    # 4: blend ink 0-40% toward white (background is already 255, so it is unchanged)
    def _lighter_ink(self, img):
        a = self._u(0.0, self.light_max)
        return (img.astype(np.float32) + a * (255.0 - img)).astype(np.uint8)

    # 7: junk, kept clear of the strokes
    def _junk(self, img):
        free = cv2.dilate((img < 200).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
        out = img.copy()
        if self.rng.random() < 0.5:                       # one printed line
            for _ in range(15):
                horizontal = self.rng.random() < 0.7
                length = int(self._u(80, 300)) if horizontal else int(self._u(40, 160))
                t = int(self.rng.integers(1, 3))
                if horizontal:
                    y, x = int(self.rng.integers(2, H - 2 - t)), int(self.rng.integers(0, W - length))
                    sl = (slice(y, y + t), slice(x, x + length))
                else:
                    x, y = int(self.rng.integers(2, W - 2 - t)), int(self.rng.integers(0, H - length))
                    sl = (slice(y, y + length), slice(x, x + t))
                if free[sl].all():
                    out[sl] = int(self._u(60, 160))
                    return out, True
            return img, False
        n, placed = int(self.rng.integers(3, 9)), 0       # a few specks
        for _ in range(40):
            if placed >= n:
                break
            y, x = int(self.rng.integers(3, H - 3)), int(self.rng.integers(3, W - 3))
            if free[y - 2:y + 3, x - 2:x + 3].all():
                cv2.circle(out, (x, y), int(self.rng.integers(1, 3)), int(self._u(40, 150)), -1)
                placed += 1
        return out, placed > 0

    # 8: mild brightness/contrast, background pushed back to white
    def _brightness_contrast(self, img):
        c, b = self._u(0.9, 1.1), self._u(-10, 10)
        f = (img.astype(np.float32) - 128.0) * c + 128.0 + b
        f = np.clip(f, 0, 255)
        f[f >= 235] = 255
        return f.astype(np.uint8)

    # 5
    def _blur(self, img):
        return cv2.GaussianBlur(img, (0, 0), self._u(*self.blur_sigma))

    # 6: mild noise and/or JPEG
    def _noise_jpeg(self, img):
        mode = self.rng.choice(["noise", "jpeg", "both"])
        if mode in ("noise", "both"):
            n = self.rng.normal(0, self._u(*self.noise_sigma), img.shape)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        if mode in ("jpeg", "both"):
            q = int(self.rng.integers(self.jpeg_q[0], self.jpeg_q[1] + 1))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return img

    def __call__(self, img, return_ops=False, index=None):
        assert img.ndim == 2 and img.dtype == np.uint8, "expected uint8 grayscale image"
        if self.seed is not None and index is not None:
            # Per-sample stream so the result depends only on (seed, epoch, index).
            self._rng = np.random.default_rng([self.seed, self.epoch, int(index)])
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        r, ops = self.rng.random, []

        do_rot, do_ss = r() < self.p_rot, r() < self.p_scaleshift
        if do_rot or do_ss:
            img = self._geometry(img, do_rot, do_ss, ops)

        u = r()                                           # 25% thick, 15% thin, 60% untouched
        if u < self.p_thick + self.p_thin:
            img = self._thickness(img, thick=u < self.p_thick)
            ops.append("thick" if u < self.p_thick else "thin")

        if r() < self.p_light:
            img = self._lighter_ink(img)
            ops.append("lighter_ink")

        if self.group2:
            if r() < self.p_junk:
                img, ok = self._junk(img)
                if ok:
                    ops.append("junk")
            if r() < self.p_bc:
                img = self._brightness_contrast(img)
                ops.append("brightness_contrast")

        if r() < self.p_blur:
            img = self._blur(img)
            ops.append("blur")
        if r() < self.p_noise_jpeg:
            img = self._noise_jpeg(img)
            ops.append("noise_jpeg")

        assert img.shape == (H, W)
        return (img, ops) if return_ops else img


# ---- hook into your Dataset (train split only) ----
# aug = SignatureAugment(group2=False) if split == "train" else None
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# if aug is not None:
#     img = aug(img)          # call separately for anchor, positive, negative