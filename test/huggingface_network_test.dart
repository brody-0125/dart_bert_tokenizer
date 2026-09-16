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
    final golden =
        jsonDecode(
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
                    'https://huggingface.co/${model['repository']}/resolve/${model['revision']}/tokenizer.json',
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
          } finally {
            client.close(force: true);
          }
        });
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
