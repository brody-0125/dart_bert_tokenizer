import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

import 'hf_fixture_support.dart';

void main() {
  const directory = 'test/fixtures/huggingface';
  final synthetic =
      jsonDecode(File('$directory/synthetic.golden.json').readAsStringSync())
          as Map<String, dynamic>;
  group('HF synthetic regression', () {
    for (final item in synthetic['cases'] as List<dynamic>) {
      final fixture = item as Map<String, dynamic>;
      test(
        fixture['name'] as String,
        () => runHfCase(
          jsonEncode(
            (synthetic['pipelines'] as List<dynamic>)[fixture['tokenizer_index']
                as int],
          ),
          fixture,
        ),
      );
    }
  });
  final manifest =
      jsonDecode(File('$directory/manifest.json').readAsStringSync())
          as List<dynamic>;
  for (final model in manifest) {
    final name = model['name'] as String;
    final raw = File('$directory/$name.reduced.json').readAsStringSync();
    final golden =
        jsonDecode(File('$directory/$name.golden.json').readAsStringSync())
            as Map<String, dynamic>;
    group('HF golden: $name', () {
      for (final item in golden['cases'] as List<dynamic>) {
        final fixture = item as Map<String, dynamic>;
        test(fixture['name'] as String, () => runHfCase(raw, fixture));
      }
    });
  }
}
