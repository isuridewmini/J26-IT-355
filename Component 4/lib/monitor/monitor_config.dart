import 'dart:convert';

/// How sensitive each state's detection is. Three threshold sets come from the
/// notebook (`thr_by_obj`), each tuned for a different metric; [custom] is the
/// school's own set.
enum Objective {
  custom('custom', 'Custom', 'Boredom 65%, Engagement 75%, Confusion 40%, Frustration 45%.'),
  balAcc('bal_acc', 'Balanced', 'Balances catching a state against false alarms.'),
  macroF1('macro_f1', 'Recommended', "The notebook's default (best average F1)."),
  acc('acc', 'Accuracy', 'Most accurate overall, but almost always reports Engaged.');

  const Objective(this.key, this.label, this.description);

  final String key;
  final String label;
  final String description;

  /// The sets that come from the notebook (and are in `web_config.json`).
  static const notebook = [balAcc, macroF1, acc];

  /// [custom] thresholds, in the model's order: Boredom, Engagement, Confusion, Frustration.
  static const customThresholds = [0.65, 0.75, 0.40, 0.45];
}

/// Everything fixed by the trained model (`web_config.json`) plus how the live
/// monitor samples and windows frames.
class MonitorConfig {
  MonitorConfig({
    required this.states,
    required this.thresholds,
    required this.defaultObjective,
    required this.featMean,
    required this.featStd,
    required this.featClip,
    required this.frames,
    this.sampleFps = 6.0,
    this.windowSeconds = 10.0,
    this.strideSeconds = 5.0,
    this.minFrames = 8,
  });

  factory MonitorConfig.fromJson(String source) {
    final j = jsonDecode(source) as Map<String, dynamic>;
    List<double> nums(Object? l) => (l as List).map((e) => (e as num).toDouble()).toList();
    final feat = j['feat'] as Map<String, dynamic>;
    final thr = j['thresholds'] as Map<String, dynamic>;
    return MonitorConfig(
      states: (j['states'] as List).cast<String>(),
      thresholds: {
        for (final o in Objective.notebook) o: nums(thr[o.key]),
        Objective.custom: Objective.customThresholds,
      },
      defaultObjective: Objective.values.firstWhere((o) => o.key == j['default_objective']),
      featMean: nums(feat['mean']),
      featStd: nums(feat['std']),
      featClip: (feat['clip'] as num).toDouble(),
      frames: j['frames'] as int,
    );
  }

  /// Boredom, Engagement, Confusion, Frustration - the model's output order.
  final List<String> states;
  final Map<Objective, List<double>> thresholds;
  final Objective defaultObjective;
  final List<double> featMean;
  final List<double> featStd;
  final double featClip;

  /// The longest sequence the model was trained on (60 frames).
  final int frames;

  /// Frames analysed per second: training used 60 frames over ~10 s.
  final double sampleFps;

  /// How much video each reading looks at (the training clip length).
  final double windowSeconds;

  /// How often a new reading is produced.
  final double strideSeconds;

  /// A window needs at least this many frames to be scored.
  final int minFrames;

  int get engagementIndex => states.indexOf('Engagement');
}
