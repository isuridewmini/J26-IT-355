import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'math_utils.dart';

/// The temporal head of the DAiSEE model, in plain Dart: 15 BiGRU networks and
/// 5 logistic-regression folds, blended `blend_w` : `1 - blend_w`.
///
/// It is the model that `head.onnx` contains (notebook cell 36, `WebHead`):
/// per-frame DINOv2 embeddings [T,1536] and standardised face features [T,16]
/// in, probabilities of Boredom / Engagement / Confusion / Frustration out.
/// Unlike the ONNX file (fixed at 60 frames) it accepts any T >= 1, exactly
/// like the notebook's `daisee_predict_video.py` does for shorter windows.
class TemporalHead {
  TemporalHead._({
    required List<_BiGruNet> nets,
    required this.nLr,
    required this.dLr,
    required this.lrMean,
    required this.lrScale,
    required this.lrW,
    required this.lrB,
    required this.thrIdx,
    required this.blendW,
    required this.dEmb,
    required this.dFeat,
    required this.hidden,
  }) : _nets = nets;

  /// [weights] is the container written by `tool/export_head_assets.py`:
  /// u32 header length, JSON header, float32 data.
  factory TemporalHead.fromBytes(Uint8List weights) {
    final headerLen = ByteData.sublistView(weights).getUint32(0, Endian.little);
    final meta = jsonDecode(utf8.decode(weights.sublist(4, 4 + headerLen))) as Map<String, dynamic>;
    final dataStart = 4 + headerLen;
    final tensors = meta['tensors'] as Map<String, dynamic>;

    Float32List get(String name) {
      final info = tensors[name] as Map<String, dynamic>;
      final count = (info['shape'] as List).cast<int>().fold<int>(1, (a, b) => a * b);
      final start = weights.offsetInBytes + dataStart + (info['offset'] as int);
      if (start % 4 == 0) return weights.buffer.asFloat32List(start, count);
      return Float32List.fromList(Uint8List.sublistView(weights, dataStart + (info['offset'] as int),
              dataStart + (info['offset'] as int) + count * 4)
          .buffer
          .asFloat32List());
    }

    final nNets = meta['n_nets'] as int;
    final nets = [
      for (var i = 0; i < nNets; i++)
        _BiGruNet({
          for (final k in _netKeys) k: get('nn$i.$k'),
        }),
    ];
    final nLr = meta['n_lr'] as int;
    final dLr = (tensors['lr.mean']!['shape'] as List).last as int;
    return TemporalHead._(
      nets: nets,
      nLr: nLr,
      dLr: dLr,
      lrMean: get('lr.mean'),
      lrScale: get('lr.scale'),
      lrW: get('lr.W'),
      lrB: get('lr.b'),
      thrIdx: (meta['thr_idx'] as List).cast<int>(),
      blendW: (meta['blend_w'] as num).toDouble(),
      dEmb: meta['d_emb'] as int,
      dFeat: meta['d_feat'] as int,
      hidden: meta['hidden'] as int,
    );
  }

  static const _netKeys = [
    'norm.weight', 'norm.bias', 'proj.1.weight', 'proj.1.bias', 'fproj.1.weight', 'fproj.1.bias',
    'gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0',
    'gru.weight_ih_l0_reverse', 'gru.weight_hh_l0_reverse', 'gru.bias_ih_l0_reverse',
    'gru.bias_hh_l0_reverse', 'att.weight', 'att.bias', 'head.1.weight', 'head.1.bias',
    'head.4.weight', 'head.4.bias',
  ];

  final List<_BiGruNet> _nets;
  final int nLr;
  final int dLr;
  final Float32List lrMean;
  final Float32List lrScale;
  final Float32List lrW; // [folds, 4, D]
  final Float32List lrB; // [folds, 4]
  final List<int> thrIdx;
  final double blendW;
  final int dEmb;
  final int dFeat;
  final int hidden;

  /// Number of output states (Boredom, Engagement, Confusion, Frustration).
  static const states = 4;

  /// [emb]: T rows of [dEmb] raw embeddings. [feat]: T rows of [dFeat]
  /// standardised features. Returns the 4 blended probabilities.
  List<double> predict(List<Float32List> emb, List<Float32List> feat) {
    final t = emb.length;
    if (t == 0 || feat.length != t) {
      throw ArgumentError('Need the same, non-zero number of embedding and feature rows.');
    }
    final pNn = List<double>.filled(states, 0);
    for (final net in _nets) {
      final p = net.run(emb, feat, thrIdx, hidden);
      for (var k = 0; k < states; k++) {
        pNn[k] += p[k] / _nets.length;
      }
    }
    final pLr = _logisticRegression(emb, feat);
    return [for (var k = 0; k < states; k++) blendW * pNn[k] + (1 - blendW) * pLr[k]];
  }

  /// Pooled features [mean, std, max of embeddings | mean, std of features],
  /// then each fold's scaler + linear layer + sigmoid, averaged over folds.
  List<double> _logisticRegression(List<Float32List> emb, List<Float32List> feat) {
    final t = emb.length;
    final d = dEmb, f = dFeat;
    final pooled = Float64List(dLr);

    for (var j = 0; j < d; j++) {
      var s = 0.0;
      var mx = double.negativeInfinity;
      for (var i = 0; i < t; i++) {
        final v = emb[i][j];
        s += v;
        if (v > mx) mx = v;
      }
      final m = s / t;
      var v2 = 0.0;
      for (var i = 0; i < t; i++) {
        final c = emb[i][j] - m;
        v2 += c * c;
      }
      pooled[j] = m;
      pooled[d + j] = math.sqrt(v2 / t + 1e-6);
      pooled[2 * d + j] = mx;
    }
    for (var j = 0; j < f; j++) {
      var s = 0.0;
      for (var i = 0; i < t; i++) {
        s += feat[i][j];
      }
      final m = s / t;
      var v2 = 0.0;
      for (var i = 0; i < t; i++) {
        final c = feat[i][j] - m;
        v2 += c * c;
      }
      pooled[3 * d + j] = m;
      pooled[3 * d + f + j] = math.sqrt(v2 / t + 1e-6);
    }

    final out = List<double>.filled(states, 0);
    for (var fold = 0; fold < nLr; fold++) {
      for (var k = 0; k < states; k++) {
        var z = lrB[fold * states + k];
        final wBase = (fold * states + k) * dLr;
        final sBase = fold * dLr;
        for (var j = 0; j < dLr; j++) {
          z += lrW[wBase + j] * ((pooled[j] - lrMean[sBase + j]) / lrScale[sBase + j]);
        }
        out[k] += sigmoid(z) / nLr;
      }
    }
    return out;
  }
}

/// One BiGRU network (the notebook's `ExportHead`).
class _BiGruNet {
  _BiGruNet(this.w);

  final Map<String, Float32List> w;

  List<double> run(List<Float32List> emb, List<Float32List> feat, List<int> thrIdx, int hidden) {
    final t = emb.length;
    final dEmb = emb[0].length, dFeat = feat[0].length;
    const fproj = 64;
    final inDim = hidden + fproj;

    // --- per-frame: LayerNorm -> Linear -> GELU, and Linear(features) -> GELU ---
    final normW = w['norm.weight']!, normB = w['norm.bias']!;
    final projW = w['proj.1.weight']!, projB = w['proj.1.bias']!;
    final fW = w['fproj.1.weight']!, fB = w['fproj.1.bias']!;
    final h = List.generate(t, (_) => Float64List(inDim));
    final xn = Float64List(dEmb);
    for (var i = 0; i < t; i++) {
      final x = emb[i];
      var mean = 0.0;
      for (var j = 0; j < dEmb; j++) {
        mean += x[j];
      }
      mean /= dEmb;
      var v = 0.0;
      for (var j = 0; j < dEmb; j++) {
        final c = x[j] - mean;
        v += c * c;
      }
      final inv = 1 / math.sqrt(v / dEmb + 1e-5);
      for (var j = 0; j < dEmb; j++) {
        xn[j] = (x[j] - mean) * inv * normW[j] + normB[j];
      }
      for (var o = 0; o < hidden; o++) {
        var s = projB[o];
        final base = o * dEmb;
        for (var j = 0; j < dEmb; j++) {
          s += projW[base + j] * xn[j];
        }
        h[i][o] = gelu(s);
      }
      final f = feat[i];
      for (var o = 0; o < fproj; o++) {
        var s = fB[o];
        for (var j = 0; j < dFeat; j++) {
          s += fW[o * dFeat + j] * f[j];
        }
        h[i][hidden + o] = gelu(s);
      }
    }

    // --- bidirectional GRU ---
    final out = List.generate(t, (_) => Float64List(2 * hidden));
    _gru(h, out, 0, hidden, inDim, '', false);
    _gru(h, out, hidden, hidden, inDim, '_reverse', true);

    // --- pooling: mean, max, attention ---
    final width = 2 * hidden;
    final mean = Float64List(width), mx = Float64List(width)..fillRange(0, width, double.negativeInfinity);
    for (final row in out) {
      for (var j = 0; j < width; j++) {
        mean[j] += row[j] / t;
        if (row[j] > mx[j]) mx[j] = row[j];
      }
    }
    final attW = w['att.weight']!, attB = w['att.bias']![0];
    final scores = Float64List(t);
    var peak = double.negativeInfinity;
    for (var i = 0; i < t; i++) {
      var s = attB;
      for (var j = 0; j < width; j++) {
        s += attW[j] * out[i][j];
      }
      scores[i] = s;
      if (s > peak) peak = s;
    }
    var denom = 0.0;
    for (var i = 0; i < t; i++) {
      scores[i] = math.exp(scores[i] - peak);
      denom += scores[i];
    }
    final att = Float64List(width);
    for (var i = 0; i < t; i++) {
      final a = scores[i] / denom;
      for (var j = 0; j < width; j++) {
        att[j] += a * out[i][j];
      }
    }

    // --- head: Linear(6H -> H) -> GELU -> Linear(H -> 16) ---
    final z = Float64List(3 * width)
      ..setRange(0, width, mean)
      ..setRange(width, 2 * width, mx)
      ..setRange(2 * width, 3 * width, att);
    final h1W = w['head.1.weight']!, h1B = w['head.1.bias']!;
    final hid = Float64List(hidden);
    for (var o = 0; o < hidden; o++) {
      var s = h1B[o];
      for (var j = 0; j < 3 * width; j++) {
        s += h1W[o * 3 * width + j] * z[j];
      }
      hid[o] = gelu(s);
    }
    final h2W = w['head.4.weight']!, h2B = w['head.4.bias']!;
    final logits = List<double>.generate(4 + 12, (o) {
      var s = h2B[o];
      for (var j = 0; j < hidden; j++) {
        s += h2W[o * hidden + j] * hid[j];
      }
      return s;
    });

    // Each state's score: mean of the binary head and the ordinal head at its level.
    return [
      for (var k = 0; k < 4; k++)
        (sigmoid(logits[k]) + sigmoid(logits[4 + k * 3 + thrIdx[k]])) / 2,
    ];
  }

  /// One direction of the GRU (PyTorch gate order r, z, n); writes its hidden
  /// states into [out] columns [col, col + hidden).
  void _gru(List<Float64List> x, List<Float64List> out, int col, int hidden, int inDim,
      String suffix, bool reverse) {
    final wih = w['gru.weight_ih_l0$suffix']!, whh = w['gru.weight_hh_l0$suffix']!;
    final bih = w['gru.bias_ih_l0$suffix']!, bhh = w['gru.bias_hh_l0$suffix']!;
    final t = x.length;
    var hPrev = Float64List(hidden);
    final gi = Float64List(3 * hidden);
    final gh = Float64List(3 * hidden);
    for (var step = 0; step < t; step++) {
      final i = reverse ? t - 1 - step : step;
      final xi = x[i];
      for (var o = 0; o < 3 * hidden; o++) {
        var s = bih[o];
        final base = o * inDim;
        for (var j = 0; j < inDim; j++) {
          s += wih[base + j] * xi[j];
        }
        gi[o] = s;
        var r = bhh[o];
        final hb = o * hidden;
        for (var j = 0; j < hidden; j++) {
          r += whh[hb + j] * hPrev[j];
        }
        gh[o] = r;
      }
      final hNew = Float64List(hidden);
      for (var j = 0; j < hidden; j++) {
        final r = sigmoid(gi[j] + gh[j]);
        final zg = sigmoid(gi[hidden + j] + gh[hidden + j]);
        final n = _tanh(gi[2 * hidden + j] + r * gh[2 * hidden + j]);
        hNew[j] = (1 - zg) * n + zg * hPrev[j];
        out[i][col + j] = hNew[j];
      }
      hPrev = hNew;
    }
  }

  static double _tanh(double x) {
    final e = math.exp(2 * x);
    return x > 20 ? 1.0 : (x < -20 ? -1.0 : (e - 1) / (e + 1));
  }
}
