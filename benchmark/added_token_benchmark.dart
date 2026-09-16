import 'dart:convert';
import 'dart:io';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main(List<String> args) {
  final raw =
      jsonDecode(
            File(
              args.isEmpty
                  ? '.dart_tool/hf-fixtures/bert-base-uncased.json'
                  : args.first,
            ).readAsStringSync(),
          )
          as Map<String, dynamic>;
  for (final count in [0, 10, 100, 1000]) {
    final pipeline = jsonDecode(jsonEncode(raw)) as Map<String, dynamic>;
    final vocab = pipeline['model']['vocab'] as Map<String, dynamic>;
    for (var i = 0; i < count; i++) {
      (pipeline['added_tokens'] as List).add({
        'content': '<extra_$i>',
        'id': vocab.length + i,
        'single_word': false,
        'lstrip': false,
        'rstrip': false,
        'normalized': false,
        'special': false,
      });
    }
    final t = WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(pipeline));
    for (final long in [false, true]) {
      final text = List.filled(
        long ? 100 : 1,
        'Hello world! café 😊 <extra_0>',
      ).join(' ');
      final iterations = long ? 20 : 1000;
      for (var i = 0; i < 100; i++) {
        t.encode(text);
      }
      final samples = <double>[];
      var checksum = 0;
      for (var round = 0; round < 5; round++) {
        final sw = Stopwatch()..start();
        for (var i = 0; i < iterations; i++) {
          checksum += t.encode(text).length;
        }
        sw.stop();
        samples.add(sw.elapsedMicroseconds / iterations);
      }
      samples.sort();
      print(
        jsonEncode({
          'count': count,
          'long': long,
          'median_us': samples[2],
          'rss_bytes': ProcessInfo.currentRss,
          'checksum': checksum,
        }),
      );
    }
  }
}
