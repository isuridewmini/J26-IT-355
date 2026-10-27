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
