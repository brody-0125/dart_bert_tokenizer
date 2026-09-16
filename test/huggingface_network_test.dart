import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

import 'hf_fixture_support.dart';

void main() {
  const directory = 'test/fixtures/huggingface';
  final manifest =
      jsonDecode(File('$directory/manifest.json').readAsStringSync())
          as List<dynamic>;
  final enabled = Platform.environment['RUN_HF_NETWORK_TESTS'] == '1';
  for (final entry in manifest) {
    final model = entry as Map<String, dynamic>;
    final golden = model.containsKey('expected_error')
        ? {'cases': <dynamic>[]}
        : jsonDecode(
                File(
                  '$directory/${model['name']}.golden.json',
                ).readAsStringSync(),
              )
              as Map<String, dynamic>;
    group(
      '${model['name']}@${model['revision']}',
      () {
        late String raw;
        setUpAll(() async {
          final client = HttpClient()
            ..connectionTimeout = const Duration(seconds: 30);
          try {
            final request = await client
                .getUrl(
                  Uri.parse(
                    'https://huggingface.co/${model['repository']}/resolve/${model['revision']}/${model['source_file'] ?? 'tokenizer.json'}',
                  ),
                )
                .timeout(const Duration(seconds: 30));
            final response = await request.close().timeout(
              const Duration(seconds: 60),
            );
            expect(response.statusCode, HttpStatus.ok);
            final bytes = await response
                .fold<List<int>>(<int>[], (all, chunk) => all..addAll(chunk))
                .timeout(const Duration(seconds: 60));
            expect(sha256.convert(bytes).toString(), model['sha256']);
            raw = utf8.decode(bytes);
            if (model['source_file'] == 'vocab.txt') {
              final pipeline =
                  jsonDecode(
                        File(
                          '$directory/${model['name']}.reduced.json',
                        ).readAsStringSync(),
                      )
                      as Map<String, dynamic>;
              final lines = const LineSplitter().convert(raw);
              pipeline['model']['vocab'] = {
                for (var i = 0; i < lines.length; i++) lines[i]: i,
              };
              raw = jsonEncode(pipeline);
            }
          } finally {
            client.close(force: true);
          }
        });
        if (model['expected_error'] case final String error) {
          test('rejects unsupported pipeline', () {
            expect(
              () => WordPieceTokenizer.fromTokenizerJsonString(raw),
              throwsA(
                isA<FormatException>().having(
                  (e) => e.message,
                  'message',
                  contains(error),
                ),
              ),
            );
          });
        }
        for (final item in golden['cases'] as List<dynamic>) {
          final fixture = item as Map<String, dynamic>;
          test(fixture['name'] as String, () => runHfCase(raw, fixture));
        }
      },
      skip: enabled
          ? false
          : 'Set RUN_HF_NETWORK_TESTS=1 to download pinned HF models',
    );
  }
}
