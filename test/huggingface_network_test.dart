import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:test/test.dart';

import 'hf_fixture_support.dart';

void main() {
  const directory = 'test/fixtures/huggingface';
  final manifest =
      jsonDecode(File('$directory/manifest.json').readAsStringSync())
          as List<dynamic>;
  final enabled = Platform.environment['RUN_HF_NETWORK_TESTS'] == '1';
  for (final entry in manifest) {
    final model = entry as Map<String, dynamic>;
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
        registerHfModelTests(model, () => raw);
      },
      skip: enabled
          ? false
          : 'Set RUN_HF_NETWORK_TESTS=1 to download pinned HF models',
    );
  }
}
