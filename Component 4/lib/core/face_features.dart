import 'dart:math' as math;

/// The 16 per-frame face features of the DAiSEE model, computed from the 468
/// face-mesh landmarks. A port of `face_feats` / `add_deltas` in
/// `docs/reference_preprocess.py` plus the standardisation in `web_config.json`.
///
/// Feature order (web_config.json `feat.names`):
/// ear_l, ear_r, mar, brow_l, brow_r, yaw, pitch, roll, cx, cy, size,
/// face_found, d_center, d_yaw, d_pitch, d_ear.
class FaceFeatures {
  FaceFeatures._();

  /// Features computed from one frame: the 11 geometric ones + face_found.
  static const baseCount = 12;

  /// Base features plus the 4 frame-to-frame motion features.
  static const totalCount = 16;

  static const landmarkCount = 468;

  /// The 11 geometric features from [xy], the 468 landmarks as pixel
  /// coordinates `[x0, y0, x1, y1, ...]` of a [width] x [height] frame.
  /// (The caller appends the face_found flag.)
  static List<double> faceFeats(List<double> xy, double width, double height) {
    if (xy.length < landmarkCount * 2) {
      throw ArgumentError('Need $landmarkCount landmarks, got ${xy.length ~/ 2}.');
    }
    double x(int i) => xy[2 * i];
    double y(int i) => xy[2 * i + 1];
    double d(int a, int b) => math.sqrt(math.pow(x(a) - x(b), 2) + math.pow(y(a) - y(b), 2));

    final iod = d(33, 263) + 1e-6;
    final fh = d(10, 152) + 1e-6;
    final earL = (d(160, 144) + d(158, 153)) / (2 * d(33, 133) + 1e-6);
    final earR = (d(385, 380) + d(387, 373)) / (2 * d(362, 263) + 1e-6);
    final mar = d(13, 14) / (d(78, 308) + 1e-6);
    final browL = d(105, 159) / iod;
    final browR = d(334, 386) / iod;
    final eyeMidX = (x(33) + x(263)) / 2;
    final eyeMidY = (y(33) + y(263)) / 2;
    final yaw = (x(1) - eyeMidX) / iod;
    final pitch = (y(1) - eyeMidY) / fh;
    final roll = math.atan2(y(263) - y(33), x(263) - x(33));

    var x0 = double.infinity, y0 = double.infinity;
    var x1 = double.negativeInfinity, y1 = double.negativeInfinity;
    for (var i = 0; i < landmarkCount; i++) {
      final px = x(i), py = y(i);
      if (px < x0) x0 = px;
      if (px > x1) x1 = px;
      if (py < y0) y0 = py;
      if (py > y1) y1 = py;
    }
    final cx = ((x0 + x1) / 2) / width;
    final cy = ((y0 + y1) / 2) / height;
    final size = math.max(x1 - x0, y1 - y0) / math.min(width, height);
    return [earL, earR, mar, browL, browR, yaw, pitch, roll, cx, cy, size];
  }

  /// [rows] are T x 12 base features (11 geometric + face_found). Returns
  /// T x 16 with the motion features appended: movement of the face centre,
  /// and the change in yaw, pitch and eye openness since the previous frame.
  /// The first row's motion is 0.
  static List<List<double>> addDeltas(List<List<double>> rows) {
    final out = <List<double>>[];
    for (var i = 0; i < rows.length; i++) {
      var dxy = 0.0, dyaw = 0.0, dpit = 0.0, dear = 0.0;
      if (i > 0) {
        final a = rows[i - 1], b = rows[i];
        dxy = math.sqrt(math.pow(b[8] - a[8], 2) + math.pow(b[9] - a[9], 2));
        dyaw = (b[5] - a[5]).abs();
        dpit = (b[6] - a[6]).abs();
        dear = ((b[0] + b[1]) / 2 - (a[0] + a[1]) / 2).abs();
      }
      out.add([...rows[i], dxy, dyaw, dpit, dear]);
    }
    return out;
  }

  /// `clip((raw - mean) / std, -clip, clip)`.
  static List<double> standardize(List<double> row, List<double> mean, List<double> std,
      {double clip = 5.0}) {
    return [
      for (var i = 0; i < row.length; i++) ((row[i] - mean[i]) / std[i]).clamp(-clip, clip).toDouble(),
    ];
  }
}
