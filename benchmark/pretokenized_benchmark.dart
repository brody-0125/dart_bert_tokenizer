import 'dart:convert';
import 'dart:io';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

Future<void> main(List<String> args) async {
  final t = WordPieceTokenizer.fromTokenizerJsonSync(args.first);
  for (final repeats in [1, 100]) {
    final words = List.generate(
      repeats,
      (_) => ['Hello,', 'world!', 'unaffordable', 'café', '😊'],
    ).expand((w) => w).toList();
    final text = words.join(' ');
    for (final mode in ['raw', 'words', 'batch', 'parallel']) {
      final batch = List.generate(8, (_) => words);
      Future<int> run() async => switch (mode) {
        'raw' => t.encode(text).length,
        'words' => t.encodePreTokenized(words).length,
        'batch' =>
          t.encodePreTokenizedBatch(batch).fold<int>(0, (n, e) => n + e.length),
        _ => (await t.encodePreTokenizedBatchParallel(
          batch,
          numWorkers: 2,
        )).fold<int>(0, (n, e) => n + e.length),
      };
      for (var i = 0; i < 20; i++) {
        await run();
      }
      final iterations = mode == 'parallel' ? 5 : (repeats == 1 ? 200 : 10);
      final samples = <double>[];
      var tokens = 0;
      for (var sample = 0; sample < 5; sample++) {
        final sw = Stopwatch()..start();
        for (var i = 0; i < iterations; i++) {
          tokens = await run();
        }
        samples.add(sw.elapsedMicroseconds / iterations);
      }
      samples.sort();
      print(
        jsonEncode({
          'mode': mode,
          'words': words.length,
          'tokens': tokens,
          'median_us': samples[2],
          'iterations': iterations,
          'sdk': Platform.version.split(' ').first,
        }),
      );
    }
  }
}
