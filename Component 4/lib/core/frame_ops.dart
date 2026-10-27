import 'dart:math' as math;
import 'dart:typed_data';

/// An RGB image: 3 bytes per pixel, row by row.
class RgbImage {
  RgbImage(this.width, this.height, this.data)
      : assert(data.length == width * height * 3);

  final int width;
  final int height;
  final Uint8List data;
}

/// The face's bounding box in image pixels.
class FaceBox {
  const FaceBox(this.x0, this.y0, this.x1, this.y1);

  final double x0, y0, x1, y1;

  /// The box around the 468 landmarks [xy] (`[x0, y0, x1, y1, ...]`, pixels).
  factory FaceBox.fromLandmarks(List<double> xy) {
    var x0 = double.infinity, y0 = double.infinity;
    var x1 = double.negativeInfinity, y1 = double.negativeInfinity;
    for (var i = 0; i + 1 < xy.length; i += 2) {
      x0 = math.min(x0, xy[i]);
      x1 = math.max(x1, xy[i]);
      y0 = math.min(y0, xy[i + 1]);
      y1 = math.max(y1, xy[i + 1]);
    }
    return FaceBox(x0, y0, x1, y1);
  }
}

/// Image preparation for the DINOv2 backbone, matching the training pipeline
/// (`crop_square` in docs/reference_preprocess.py): a square crop around the
/// face with a 1.3x margin, resized to 224x224 with OpenCV's INTER_AREA
/// (exact area averaging).
class FrameOps {
  FrameOps._();

  static const imageSize = 224;

  /// Decodes an Android NV21 frame (a Y plane of [width] x [height] bytes, then
  /// interleaved V/U at half resolution) to RGB with the full-range BT.601
  /// conversion Android itself uses.
  static RgbImage nv21ToRgb(Uint8List nv21, int width, int height) {
    if (nv21.length < width * height * 3 ~/ 2) {
      throw ArgumentError('NV21 buffer too small for ${width}x$height.');
    }
    final out = Uint8List(width * height * 3);
    final uvStart = width * height;
    var o = 0;
    for (var y = 0; y < height; y++) {
      final uvRow = uvStart + (y >> 1) * width;
      for (var x = 0; x < width; x++) {
        final yy = nv21[y * width + x] << 16;
        final uvIndex = uvRow + (x & ~1);
        final v = nv21[uvIndex] - 128;
        final u = nv21[uvIndex + 1] - 128;
        out[o++] = _clamp((yy + 91881 * v + 32768) >> 16);
        out[o++] = _clamp((yy - 22554 * u - 46802 * v + 32768) >> 16);
        out[o++] = _clamp((yy + 116130 * u + 32768) >> 16);
      }
    }
    return RgbImage(width, height, out);
  }

  static int _clamp(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

  /// Rotates [img] clockwise by [degrees] (0, 90, 180 or 270).
  static RgbImage rotateClockwise(RgbImage img, int degrees) {
    final d = ((degrees % 360) + 360) % 360;
    if (d == 0) return img;
    if (d % 90 != 0) throw ArgumentError('Rotation must be a multiple of 90 degrees.');
    final w = img.width, h = img.height;
    final nw = d == 180 ? w : h;
    final nh = d == 180 ? h : w;
    final out = Uint8List(nw * nh * 3);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final int nx, ny;
        switch (d) {
          case 90:
            nx = h - 1 - y;
            ny = x;
          case 180:
            nx = w - 1 - x;
            ny = h - 1 - y;
          default: // 270
            nx = y;
            ny = w - 1 - x;
        }
        final s = (y * w + x) * 3, t = (ny * nw + nx) * 3;
        out[t] = img.data[s];
        out[t + 1] = img.data[s + 1];
        out[t + 2] = img.data[s + 2];
      }
    }
    return RgbImage(nw, nh, out);
  }

  /// The crop rectangle `crop_square` uses: a square around [box] scaled by
  /// [margin] (at least 32 px), or the central 80% square with no box,
  /// clipped to the frame. Returns (x0, y0, x1, y1), end-exclusive.
  static (int, int, int, int) cropWindow(int width, int height, FaceBox? box, {double margin = 1.3}) {
    int s, cx, cy;
    if (box == null) {
      s = (0.8 * math.min(height, width)).toInt();
      cx = width ~/ 2;
      cy = height ~/ 2;
    } else {
      s = (math.max(box.x1 - box.x0, box.y1 - box.y0) * margin).toInt();
      cx = ((box.x0 + box.x1) / 2).toInt();
      cy = ((box.y0 + box.y1) / 2).toInt();
    }
    s = math.max(s, 32);
    final xa = math.max(0, cx - s ~/ 2), ya = math.max(0, cy - s ~/ 2);
    final xb = math.min(width, cx + s ~/ 2), yb = math.min(height, cy + s ~/ 2);
    if (xb <= xa || yb <= ya) return (0, 0, width, height); // nothing left: use the whole frame
    return (xa, ya, xb, yb);
  }

  /// The 224x224 RGB face crop (HWC, bytes), as the notebook's `crop_square`.
  static Uint8List cropFaceRgb(RgbImage frame, FaceBox? box, {double margin = 1.3}) {
    final (xa, ya, xb, yb) = cropWindow(frame.width, frame.height, box, margin: margin);
    return areaResize(frame, xa, ya, xb - xa, yb - ya, imageSize, imageSize);
  }

  /// The backbone input for one frame: the face crop as float32 CHW in 0-1.
  static Float32List cropFaceTensor(RgbImage frame, FaceBox? box, {double margin = 1.3}) {
    return rgbToChw(cropFaceRgb(frame, box, margin: margin), imageSize, imageSize);
  }

  /// HWC bytes -> float32 CHW scaled to 0-1.
  static Float32List rgbToChw(Uint8List hwc, int width, int height) {
    final plane = width * height;
    final out = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      out[i] = hwc[i * 3] / 255.0;
      out[plane + i] = hwc[i * 3 + 1] / 255.0;
      out[2 * plane + i] = hwc[i * 3 + 2] / 255.0;
    }
    return out;
  }

  /// Resizes the window ([x0], [y0], [w] x [h]) of [src] to [outW] x [outH] by
  /// exact area averaging (OpenCV `INTER_AREA`), rounding to bytes.
  static Uint8List areaResize(RgbImage src, int x0, int y0, int w, int h, int outW, int outH) {
    final wx = _AxisWeights(w, outW), wy = _AxisWeights(h, outH);

    // Horizontal pass: h rows x outW columns x 3 channels.
    final mid = Float32List(h * outW * 3);
    for (var y = 0; y < h; y++) {
      final rowBase = ((y0 + y) * src.width + x0) * 3;
      for (var dx = 0; dx < outW; dx++) {
        var r = 0.0, g = 0.0, b = 0.0;
        for (var k = wx.start[dx]; k < wx.end[dx]; k++) {
          final wgt = wx.weight[k];
          final s = rowBase + wx.index[k] * 3;
          r += src.data[s] * wgt;
          g += src.data[s + 1] * wgt;
          b += src.data[s + 2] * wgt;
        }
        final m = (y * outW + dx) * 3;
        mid[m] = r;
        mid[m + 1] = g;
        mid[m + 2] = b;
      }
    }

    // Vertical pass.
    final out = Uint8List(outW * outH * 3);
    for (var dy = 0; dy < outH; dy++) {
      for (var dx = 0; dx < outW; dx++) {
        var r = 0.0, g = 0.0, b = 0.0;
        for (var k = wy.start[dy]; k < wy.end[dy]; k++) {
          final wgt = wy.weight[k];
          final m = (wy.index[k] * outW + dx) * 3;
          r += mid[m] * wgt;
          g += mid[m + 1] * wgt;
          b += mid[m + 2] * wgt;
        }
        final o = (dy * outW + dx) * 3;
        out[o] = _clamp((r + 0.5).floor());
        out[o + 1] = _clamp((g + 0.5).floor());
        out[o + 2] = _clamp((b + 0.5).floor());
      }
    }
    return out;
  }
}

/// For each output index, the input pixels it covers and how much of each
/// (the fraction of the output pixel's footprint), for one axis.
class _AxisWeights {
  _AxisWeights(int nIn, int nOut)
      : start = Int32List(nOut),
        end = Int32List(nOut) {
    final scale = nIn / nOut;
    final idx = <int>[], wts = <double>[];
    for (var d = 0; d < nOut; d++) {
      final a = d * scale, b = (d + 1) * scale;
      start[d] = idx.length;
      for (var i = a.floor(); i < math.min(nIn, b.ceil()); i++) {
        final cover = math.min(b, i + 1.0) - math.max(a, i.toDouble());
        if (cover > 0) {
          idx.add(i);
          wts.add(cover / scale);
        }
      }
      end[d] = idx.length;
    }
    index = Int32List.fromList(idx);
    weight = Float64List.fromList(wts);
  }

  final Int32List start;
  final Int32List end;
  late final Int32List index;
  late final Float64List weight;
}
