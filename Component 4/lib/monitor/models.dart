import 'dart:typed_data';

/// One analysed camera frame: when it was taken, the DINOv2 embedding of the
/// face crop, and the 12 base face features (the last is the face-found flag).
class FrameSample {
  FrameSample({required this.t, required this.embedding, required this.baseFeatures});

  /// Seconds since monitoring started.
  final double t;
  final Float32List embedding;
  final List<double> baseFeatures;

  bool get faceFound => baseFeatures.last > 0.5;
}

/// A state's probability next to its decision threshold.
class StateReading {
  const StateReading({
    required this.name,
    required this.probability,
    required this.threshold,
  });

  final String name;
  final double probability;
  final double threshold;

  bool get present => probability >= threshold;

  /// probability / threshold: how far past (or short of) the decision line.
  double get ratio => probability / threshold;
}

/// One monitoring result: the model's view of the last ~10 seconds.
class MonitorReading {
  const MonitorReading({
    required this.time,
    required this.states,
    required this.label,
    required this.faceVisibility,
    required this.framesUsed,
  });

  /// Seconds since monitoring started when this reading was made.
  final double time;

  /// Boredom, Engagement, Confusion, Frustration (model order).
  final List<StateReading> states;

  /// Plain-language summary, e.g. "Engaged, Confused".
  final String label;

  /// Fraction of the window's frames in which a face was found (0-1).
  final double faceVisibility;

  /// Frames the model saw (it was trained on 60).
  final int framesUsed;

  StateReading state(String name) => states.firstWhere((s) => s.name == name);

  bool get engaged => state('Engagement').present;
}

/// Something the teacher should look at.
enum AlertKind { lowEngagement, boredom, confusion, frustration, faceNotVisible }

class MonitorAlert {
  const MonitorAlert({required this.kind, required this.since, required this.message});

  final AlertKind kind;

  /// Time (seconds) the condition first appeared.
  final double since;
  final String message;
}
