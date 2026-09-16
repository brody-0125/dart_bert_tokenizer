import 'dart:io';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() async {
  print('=' * 70);
  print('HuggingFace BERT Tokenizer Compatibility Benchmark');
  print('=' * 70);
  print('');

  final vocabPath = _findVocabFile();
  if (vocabPath == null) {
    print('ERROR: vocab.txt not found.');
    print(
      'Please ensure vocab.txt is in the project root or benchmark directory.',
    );
    exit(1);
  }

  print('Loading vocabulary from: $vocabPath');
  final sw = Stopwatch()..start();
  final tokenizer = WordPieceTokenizer.fromVocabFileSync(vocabPath);
  sw.stop();
  print(
    'Vocabulary loaded: ${tokenizer.vocab.size} tokens in ${sw.elapsedMilliseconds}ms',
  );
  print('');

  final results = BenchmarkResults();

  await _runSingleEncodingTests(tokenizer, results);
  await _runPairEncodingTests(tokenizer, results);
  await _runEdgeCaseTests(tokenizer, results);
  await _runOffsetMappingTests(tokenizer, results);
  await _runPerformanceTest(tokenizer, results);

  _printSummary(results);

  if (results.failures.isNotEmpty) {
    exit(1);
  }
}

String? _findVocabFile() {
  final paths = ['vocab.txt', 'benchmark/vocab.txt', '../vocab.txt'];
  for (final path in paths) {
    if (File(path).existsSync()) return path;
  }
  return null;
}

class BenchmarkResults {
  int passed = 0;
  int failed = 0;
  final List<String> failures = [];
  double? tokensPerSecond;

  void pass(String testName) {
    passed++;
    print('  [PASS] $testName');
  }

  void fail(String testName, String reason) {
    failed++;
    failures.add('$testName: $reason');
    print('  [FAIL] $testName');
    print('         $reason');
  }
}

Future<void> _runSingleEncodingTests(
  WordPieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('1. SINGLE ENCODING COMPATIBILITY');
  print('-' * 70);

  for (final tc in _singleEncodingTestCases) {
    final encoding = tokenizer.encode(tc.input);

    final tokensMatch = _listEquals(
      encoding.tokens.toList(),
      tc.expectedTokens,
    );
    final idsMatch = _listEquals(encoding.ids.toList(), tc.expectedIds);
    final typeIdsMatch = _listEquals(
      encoding.typeIds.toList(),
      tc.expectedTypeIds,
    );
    final attentionMatch = _listEquals(
      encoding.attentionMask.toList(),
      tc.expectedAttentionMask,
    );

    if (tokensMatch && idsMatch && typeIdsMatch && attentionMatch) {
      results.pass(tc.name);
    } else {
      final reasons = <String>[];
      if (!tokensMatch) {
        reasons.add(
          'tokens: expected ${tc.expectedTokens}, got ${encoding.tokens}',
        );
      }
      if (!idsMatch) {
        reasons.add('ids: expected ${tc.expectedIds}, got ${encoding.ids}');
      }
      if (!typeIdsMatch) {
        reasons.add(
          'typeIds: expected ${tc.expectedTypeIds}, got ${encoding.typeIds}',
        );
      }
      if (!attentionMatch) {
        reasons.add(
          'attention: expected ${tc.expectedAttentionMask}, got ${encoding.attentionMask}',
        );
      }
      results.fail(tc.name, reasons.join('; '));
    }
  }
  print('');
}

Future<void> _runPairEncodingTests(
  WordPieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('2. PAIR ENCODING COMPATIBILITY');
  print('-' * 70);

  for (final tc in _pairEncodingTestCases) {
    final encoding = tokenizer.encodePair(tc.inputA, tc.inputB);

    final tokensMatch = _listEquals(
      encoding.tokens.toList(),
      tc.expectedTokens,
    );
    final idsMatch = _listEquals(encoding.ids.toList(), tc.expectedIds);
    final typeIdsMatch = _listEquals(
      encoding.typeIds.toList(),
      tc.expectedTypeIds,
    );

    if (tokensMatch && idsMatch && typeIdsMatch) {
      results.pass(tc.name);
    } else {
      final reasons = <String>[];
      if (!tokensMatch) {
        reasons.add(
          'tokens: expected ${tc.expectedTokens}, got ${encoding.tokens}',
        );
      }
      if (!idsMatch) {
        reasons.add('ids: expected ${tc.expectedIds}, got ${encoding.ids}');
      }
      if (!typeIdsMatch) {
        reasons.add(
          'typeIds: expected ${tc.expectedTypeIds}, got ${encoding.typeIds}',
        );
      }
      results.fail(tc.name, reasons.join('; '));
    }
  }
  print('');
}

Future<void> _runEdgeCaseTests(
  WordPieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('3. EDGE CASES COMPATIBILITY');
  print('-' * 70);

  for (final tc in _edgeCaseTestCases) {
    final encoding = tokenizer.encode(tc.input);

    final tokensMatch = _listEquals(
      encoding.tokens.toList(),
      tc.expectedTokens,
    );
    final idsMatch = _listEquals(encoding.ids.toList(), tc.expectedIds);

    if (tokensMatch && idsMatch) {
      results.pass(tc.name);
    } else {
      final reasons = <String>[];
      if (!tokensMatch) {
        reasons.add(
          'tokens: expected ${tc.expectedTokens}, got ${encoding.tokens}',
        );
      }
      if (!idsMatch) {
        reasons.add('ids: expected ${tc.expectedIds}, got ${encoding.ids}');
      }
      results.fail(tc.name, reasons.join('; '));
    }
  }
  print('');
}

Future<void> _runOffsetMappingTests(
  WordPieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('4. OFFSET MAPPING COMPATIBILITY');
  print('-' * 70);

  for (final tc in _offsetTestCases) {
    final encoding = tokenizer.encode(tc.input);

    final offsetsMatch = _offsetsEqual(encoding.offsets, tc.expectedOffsets);

    if (offsetsMatch) {
      results.pass(tc.name);
    } else {
      results.fail(
        tc.name,
        'offsets: expected ${tc.expectedOffsets}, got ${encoding.offsets}',
      );
    }
  }
  print('');
}

Future<void> _runPerformanceTest(
  WordPieceTokenizer tokenizer,
  BenchmarkResults results,
) async {
  print('-' * 70);
  print('5. PERFORMANCE BENCHMARK');
  print('-' * 70);

  final texts = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is a subset of artificial intelligence.',
    'Natural language processing enables computers to understand human language.',
    'Transformers have revolutionized the field of NLP.',
  ];

  for (var i = 0; i < 100; i++) {
    for (final t in texts) {
      tokenizer.encode(t);
    }
  }

  const iterations = 1000;
  final allTexts = <String>[];
  for (var i = 0; i < iterations; i++) {
    allTexts.addAll(texts);
  }

  final sw = Stopwatch()..start();
  var totalTokens = 0;
  for (final text in allTexts) {
    final encoding = tokenizer.encode(text);
    totalTokens += encoding.length;
  }
  sw.stop();

  final tokensPerSec = totalTokens / sw.elapsedMilliseconds * 1000;
  results.tokensPerSecond = tokensPerSec;

  print('  Texts encoded: ${allTexts.length}');
  print('  Total tokens: $totalTokens');
  print('  Time: ${sw.elapsedMilliseconds}ms');
  print('  Throughput: ${_formatNumber(tokensPerSec.round())} tokens/sec');

  if (tokensPerSec >= 500000) {
    results.pass('Throughput >= 500K tokens/sec');
  } else {
    results.fail(
      'Throughput >= 500K tokens/sec',
      'Got ${_formatNumber(tokensPerSec.round())} tokens/sec',
    );
  }
  print('');
}

void _printSummary(BenchmarkResults results) {
  print('=' * 70);
  print('SUMMARY');
  print('=' * 70);
  print('Total tests: ${results.passed + results.failed}');
  print('Passed: ${results.passed}');
  print('Failed: ${results.failed}');

  final accuracy = (results.passed / (results.passed + results.failed) * 100)
      .toStringAsFixed(1);
  print('Accuracy: $accuracy%');

  if (results.tokensPerSecond != null) {
    print(
      'Throughput: ${_formatNumber(results.tokensPerSecond!.round())} tokens/sec',
    );
  }

  if (results.failures.isNotEmpty) {
    print('');
    print('FAILURES:');
    for (final f in results.failures) {
      print('  - $f');
    }
  }

  print('');
  if (results.failed == 0) {
    print('HuggingFace compatibility: VERIFIED');
  } else {
    print('HuggingFace compatibility: FAILED');
  }
}

bool _listEquals<T>(List<T> a, List<T> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

bool _offsetsEqual(List<(int, int)> a, List<(int, int)> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i].$1 != b[i].$1 || a[i].$2 != b[i].$2) return false;
  }
  return true;
}

String _formatNumber(int n) {
  if (n >= 1000000) {
    return '${(n / 1000000).toStringAsFixed(2)}M';
  } else if (n >= 1000) {
    return '${(n / 1000).toStringAsFixed(0)}K';
  }
  return n.toString();
}

class SingleEncodingTestCase {
  final String name;
  final String input;
  final List<String> expectedTokens;
  final List<int> expectedIds;
  final List<int> expectedTypeIds;
  final List<int> expectedAttentionMask;

  const SingleEncodingTestCase({
    required this.name,
    required this.input,
    required this.expectedTokens,
    required this.expectedIds,
    required this.expectedTypeIds,
    required this.expectedAttentionMask,
  });
}

class PairEncodingTestCase {
  final String name;
  final String inputA;
  final String inputB;
  final List<String> expectedTokens;
  final List<int> expectedIds;
  final List<int> expectedTypeIds;

  const PairEncodingTestCase({
    required this.name,
    required this.inputA,
    required this.inputB,
    required this.expectedTokens,
    required this.expectedIds,
    required this.expectedTypeIds,
  });
}

class EdgeCaseTestCase {
  final String name;
  final String input;
  final List<String> expectedTokens;
  final List<int> expectedIds;

  const EdgeCaseTestCase({
    required this.name,
    required this.input,
    required this.expectedTokens,
    required this.expectedIds,
  });
}

class OffsetTestCase {
  final String name;
  final String input;
  final List<(int, int)> expectedOffsets;

  const OffsetTestCase({
    required this.name,
    required this.input,
    required this.expectedOffsets,
  });
}

// =============================================================================
// TEST DATA (67 single encoding + 3 pair encoding + 12 edge cases + 3 offset)
// Generated from HuggingFace BertWordPieceTokenizer with vocab.txt
// =============================================================================

const _singleEncodingTestCases = [
  SingleEncodingTestCase(
    name: 'Simple greeting',
    input: 'Hello, world!',
    expectedTokens: ['[CLS]', 'hello', ',', 'world', '!', '[SEP]'],
    expectedIds: [101, 7592, 1010, 2088, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Classic pangram',
    input: 'The quick brown fox jumps over the lazy dog.',
    expectedTokens: [
      '[CLS]',
      'the',
      'quick',
      'brown',
      'fox',
      'jumps',
      'over',
      'the',
      'lazy',
      'dog',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      1996,
      4248,
      2829,
      4419,
      14523,
      2058,
      1996,
      13971,
      3899,
      1012,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Subword tokenization',
    input: 'tokenization',
    expectedTokens: ['[CLS]', 'token', '##ization', '[SEP]'],
    expectedIds: [101, 19204, 3989, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Complex subword',
    input: 'unbelievable',
    expectedTokens: ['[CLS]', 'unbelievable', '[SEP]'],
    expectedIds: [101, 23653, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Accented text (cafe)',
    input: 'cafe',
    expectedTokens: ['[CLS]', 'cafe', '[SEP]'],
    expectedIds: [101, 7668, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Accented text (naive)',
    input: 'naive',
    expectedTokens: ['[CLS]', 'naive', '[SEP]'],
    expectedIds: [101, 15743, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Accented text (resume)',
    input: 'resume',
    expectedTokens: ['[CLS]', 'resume', '[SEP]'],
    expectedIds: [101, 13746, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Numbers',
    input: '12345',
    expectedTokens: ['[CLS]', '123', '##45', '[SEP]'],
    expectedIds: [101, 13138, 19961, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Multiple punctuation',
    input: '!!??',
    expectedTokens: ['[CLS]', '!', '!', '?', '?', '[SEP]'],
    expectedIds: [101, 999, 999, 1029, 1029, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'BERT model name',
    input: 'BERT is a transformer-based model.',
    expectedTokens: [
      '[CLS]',
      'bert',
      'is',
      'a',
      'transform',
      '##er',
      '-',
      'based',
      'model',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      14324,
      2003,
      1037,
      10938,
      2121,
      1011,
      2241,
      2944,
      1012,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'NLP text',
    input: 'I love NLP!',
    expectedTokens: ['[CLS]', 'i', 'love', 'nl', '##p', '!', '[SEP]'],
    expectedIds: [101, 1045, 2293, 17953, 2361, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (I'm)",
    input: "I'm happy",
    expectedTokens: ['[CLS]', 'i', "'", 'm', 'happy', '[SEP]'],
    expectedIds: [101, 1045, 1005, 1049, 3407, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (don't)",
    input: "don't",
    expectedTokens: ['[CLS]', 'don', "'", 't', '[SEP]'],
    expectedIds: [101, 2123, 1005, 1056, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Hyphenated word',
    input: 'state-of-the-art',
    expectedTokens: [
      '[CLS]',
      'state',
      '-',
      'of',
      '-',
      'the',
      '-',
      'art',
      '[SEP]',
    ],
    expectedIds: [101, 2110, 1011, 1997, 1011, 1996, 1011, 2396, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Machine learning sentence',
    input: 'Machine learning is amazing!',
    expectedTokens: [
      '[CLS]',
      'machine',
      'learning',
      'is',
      'amazing',
      '!',
      '[SEP]',
    ],
    expectedIds: [101, 3698, 4083, 2003, 6429, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Subword (preprocessing)',
    input: 'preprocessing',
    expectedTokens: ['[CLS]', 'prep', '##ro', '##ces', '##sing', '[SEP]'],
    expectedIds: [101, 17463, 3217, 9623, 7741, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Subword (playing)',
    input: 'playing',
    expectedTokens: ['[CLS]', 'playing', '[SEP]'],
    expectedIds: [101, 2652, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Subword (transformers)',
    input: 'transformers',
    expectedTokens: ['[CLS]', 'transformers', '[SEP]'],
    expectedIds: [101, 19081, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Question sentence',
    input: 'Hello, how are you?',
    expectedTokens: ['[CLS]', 'hello', ',', 'how', 'are', 'you', '?', '[SEP]'],
    expectedIds: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Ellipsis in text',
    input: 'test...test',
    expectedTokens: ['[CLS]', 'test', '.', '.', '.', 'test', '[SEP]'],
    expectedIds: [101, 3231, 1012, 1012, 1012, 3231, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Year in sentence',
    input: 'The year is 2024.',
    expectedTokens: ['[CLS]', 'the', 'year', 'is', '202', '##4', '.', '[SEP]'],
    expectedIds: [101, 1996, 2095, 2003, 16798, 2549, 1012, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Framework names',
    input: 'TensorFlow and PyTorch are popular.',
    expectedTokens: [
      '[CLS]',
      'tensor',
      '##flow',
      'and',
      'p',
      '##yt',
      '##or',
      '##ch',
      'are',
      'popular',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      23435,
      12314,
      1998,
      1052,
      22123,
      2953,
      2818,
      2024,
      2759,
      1012,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Long NLP sentence',
    input:
        'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.',
    expectedTokens: [
      '[CLS]',
      'natural',
      'language',
      'processing',
      '(',
      'nl',
      '##p',
      ')',
      'is',
      'a',
      'sub',
      '##field',
      'of',
      'linguistics',
      ',',
      'computer',
      'science',
      ',',
      'and',
      'artificial',
      'intelligence',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      3019,
      2653,
      6364,
      1006,
      17953,
      2361,
      1007,
      2003,
      1037,
      4942,
      3790,
      1997,
      15397,
      1010,
      3274,
      2671,
      1010,
      1998,
      7976,
      4454,
      1012,
      102,
    ],
    expectedTypeIds: [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ],
    expectedAttentionMask: [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
    ],
  ),
  SingleEncodingTestCase(
    name: 'Mixed symbols',
    input: r'Hello @user #hashtag $100 50%',
    expectedTokens: [
      '[CLS]',
      'hello',
      '@',
      'user',
      '#',
      'hash',
      '##tag',
      r'$',
      '100',
      '50',
      '%',
      '[SEP]',
    ],
    expectedIds: [
      101,
      7592,
      1030,
      5310,
      1001,
      23325,
      15900,
      1002,
      2531,
      2753,
      1003,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Email in sentence',
    input: 'Email: test@example.com',
    expectedTokens: [
      '[CLS]',
      'email',
      ':',
      'test',
      '@',
      'example',
      '.',
      'com',
      '[SEP]',
    ],
    expectedIds: [101, 10373, 1024, 3231, 1030, 2742, 1012, 4012, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Complex URL',
    input: 'https://www.example.com/path?query=value',
    expectedTokens: [
      '[CLS]',
      'https',
      ':',
      '/',
      '/',
      'www',
      '.',
      'example',
      '.',
      'com',
      '/',
      'path',
      '?',
      'query',
      '=',
      'value',
      '[SEP]',
    ],
    expectedIds: [
      101,
      16770,
      1024,
      1013,
      1013,
      7479,
      1012,
      2742,
      1012,
      4012,
      1013,
      4130,
      1029,
      23032,
      1027,
      3643,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'German umlaut (Zurich)',
    input: 'Zurich',
    expectedTokens: ['[CLS]', 'zurich', '[SEP]'],
    expectedIds: [101, 10204, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Portuguese (Sao Paulo)',
    input: 'Sao Paulo',
    expectedTokens: ['[CLS]', 'sao', 'paulo', '[SEP]'],
    expectedIds: [101, 7509, 9094, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'German umlaut (Munchen)',
    input: 'Munchen',
    expectedTokens: ['[CLS]', 'munchen', '[SEP]'],
    expectedIds: [101, 25802, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Multiple accented words',
    input: 'naive cafe resume',
    expectedTokens: ['[CLS]', 'naive', 'cafe', 'resume', '[SEP]'],
    expectedIds: [101, 15743, 7668, 13746, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'French accent (fiancee)',
    input: 'fiancee',
    expectedTokens: ['[CLS]', 'fiancee', '[SEP]'],
    expectedIds: [101, 19455, 102],
    expectedTypeIds: [0, 0, 0],
    expectedAttentionMask: [1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Spanish tilde (pinata)',
    input: 'pinata',
    expectedTokens: ['[CLS]', 'pin', '##ata', '[SEP]'],
    expectedIds: [101, 9231, 6790, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Very long word 1',
    input: 'antidisestablishmentarianism',
    expectedTokens: [
      '[CLS]',
      'anti',
      '##dis',
      '##est',
      '##ab',
      '##lish',
      '##ment',
      '##arian',
      '##ism',
      '[SEP]',
    ],
    expectedIds: [101, 3424, 10521, 4355, 7875, 13602, 3672, 12199, 2964, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Very long word 2',
    input: 'supercalifragilisticexpialidocious',
    expectedTokens: [
      '[CLS]',
      'super',
      '##cal',
      '##if',
      '##rag',
      '##ilis',
      '##tic',
      '##ex',
      '##pia',
      '##lid',
      '##oc',
      '##ious',
      '[SEP]',
    ],
    expectedIds: [
      101,
      3565,
      9289,
      10128,
      29181,
      24411,
      4588,
      10288,
      19312,
      21273,
      10085,
      6313,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Very long word 3',
    input: 'internationalization',
    expectedTokens: ['[CLS]', 'international', '##ization', '[SEP]'],
    expectedIds: [101, 2248, 3989, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Scientific term',
    input: 'deoxyribonucleic',
    expectedTokens: [
      '[CLS]',
      'de',
      '##ox',
      '##yr',
      '##ib',
      '##on',
      '##uc',
      '##lei',
      '##c',
      '[SEP]',
    ],
    expectedIds: [
      101,
      2139,
      11636,
      12541,
      12322,
      2239,
      14194,
      23057,
      2278,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Numbers in sentence',
    input: 'I have 3 cats and 2 dogs.',
    expectedTokens: [
      '[CLS]',
      'i',
      'have',
      '3',
      'cats',
      'and',
      '2',
      'dogs',
      '.',
      '[SEP]',
    ],
    expectedIds: [101, 1045, 2031, 1017, 8870, 1998, 1016, 6077, 1012, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Temperature with degree',
    input: 'The temperature is 72.5F.',
    expectedTokens: [
      '[CLS]',
      'the',
      'temperature',
      'is',
      '72',
      '.',
      '5',
      '##f',
      '.',
      '[SEP]',
    ],
    expectedIds: [101, 1996, 4860, 2003, 5824, 1012, 1019, 2546, 1012, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Date format',
    input: '2023-12-25',
    expectedTokens: ['[CLS]', '202', '##3', '-', '12', '-', '25', '[SEP]'],
    expectedIds: [101, 16798, 2509, 1011, 2260, 1011, 2423, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Time format',
    input: '10:30 AM',
    expectedTokens: ['[CLS]', '10', ':', '30', 'am', '[SEP]'],
    expectedIds: [101, 2184, 1024, 2382, 2572, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Formatted number',
    input: '1,234,567.89',
    expectedTokens: [
      '[CLS]',
      '1',
      ',',
      '234',
      ',',
      '56',
      '##7',
      '.',
      '89',
      '[SEP]',
    ],
    expectedIds: [101, 1015, 1010, 22018, 1010, 5179, 2581, 1012, 6486, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (won't)",
    input: "won't",
    expectedTokens: ['[CLS]', 'won', "'", 't', '[SEP]'],
    expectedIds: [101, 2180, 1005, 1056, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (it's)",
    input: "it's",
    expectedTokens: ['[CLS]', 'it', "'", 's', '[SEP]'],
    expectedIds: [101, 2009, 1005, 1055, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (they're)",
    input: "they're",
    expectedTokens: ['[CLS]', 'they', "'", 're', '[SEP]'],
    expectedIds: [101, 2027, 1005, 2128, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: "Contraction (we've)",
    input: "we've",
    expectedTokens: ['[CLS]', 'we', "'", 've', '[SEP]'],
    expectedIds: [101, 2057, 1005, 2310, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Hyphenated (well-known)',
    input: 'well-known',
    expectedTokens: ['[CLS]', 'well', '-', 'known', '[SEP]'],
    expectedIds: [101, 2092, 1011, 2124, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Hyphenated (self-driving)',
    input: 'self-driving',
    expectedTokens: ['[CLS]', 'self', '-', 'driving', '[SEP]'],
    expectedIds: [101, 2969, 1011, 4439, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Hyphenated (twenty-first)',
    input: 'twenty-first',
    expectedTokens: ['[CLS]', 'twenty', '-', 'first', '[SEP]'],
    expectedIds: [101, 3174, 1011, 2034, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Exclamation with question',
    input: 'Really?!',
    expectedTokens: ['[CLS]', 'really', '?', '!', '[SEP]'],
    expectedIds: [101, 2428, 1029, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Trailing ellipsis',
    input: 'Wait...',
    expectedTokens: ['[CLS]', 'wait', '.', '.', '.', '[SEP]'],
    expectedIds: [101, 3524, 1012, 1012, 1012, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Multiple exclamations',
    input: 'Hello!!!',
    expectedTokens: ['[CLS]', 'hello', '!', '!', '!', '[SEP]'],
    expectedIds: [101, 7592, 999, 999, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Mixed punctuation',
    input: 'What?!?!',
    expectedTokens: ['[CLS]', 'what', '?', '!', '?', '!', '[SEP]'],
    expectedIds: [101, 2054, 1029, 999, 1029, 999, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Double quoted',
    input: '"Hello"',
    expectedTokens: ['[CLS]', '"', 'hello', '"', '[SEP]'],
    expectedIds: [101, 1000, 7592, 1000, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Single quoted',
    input: "'Hello'",
    expectedTokens: ['[CLS]', "'", 'hello', "'", '[SEP]'],
    expectedIds: [101, 1005, 7592, 1005, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Quote in sentence',
    input: "He said 'hello'",
    expectedTokens: ['[CLS]', 'he', 'said', "'", 'hello', "'", '[SEP]'],
    expectedIds: [101, 2002, 2056, 1005, 7592, 1005, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Underscore identifier',
    input: 'function_name',
    expectedTokens: ['[CLS]', 'function', '_', 'name', '[SEP]'],
    expectedIds: [101, 3853, 1035, 2171, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'camelCase',
    input: 'camelCase',
    expectedTokens: ['[CLS]', 'camel', '##case', '[SEP]'],
    expectedIds: [101, 19130, 18382, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'PascalCase',
    input: 'PascalCase',
    expectedTokens: ['[CLS]', 'pascal', '##case', '[SEP]'],
    expectedIds: [101, 17878, 18382, 102],
    expectedTypeIds: [0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'snake_case',
    input: 'snake_case',
    expectedTokens: ['[CLS]', 'snake', '_', 'case', '[SEP]'],
    expectedIds: [101, 7488, 1035, 2553, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'SCREAMING_CASE',
    input: 'SCREAMING_CASE',
    expectedTokens: ['[CLS]', 'screaming', '_', 'case', '[SEP]'],
    expectedIds: [101, 7491, 1035, 2553, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Greek letter beta',
    input: 'b-testing',
    expectedTokens: ['[CLS]', 'b', '-', 'testing', '[SEP]'],
    expectedIds: [101, 1038, 1011, 5604, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Greek letter alpha',
    input: 'a-version',
    expectedTokens: ['[CLS]', 'a', '-', 'version', '[SEP]'],
    expectedIds: [101, 1037, 1011, 2544, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Greek letter mu',
    input: 'u-controller',
    expectedTokens: ['[CLS]', 'u', '-', 'controller', '[SEP]'],
    expectedIds: [101, 1057, 1011, 11486, 102],
    expectedTypeIds: [0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Repeated character',
    input: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    expectedTokens: [
      '[CLS]',
      'aaa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##aa',
      '##a',
      '[SEP]',
    ],
    expectedIds: [
      101,
      13360,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      11057,
      2050,
      102,
    ],
    expectedTypeIds: [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ],
    expectedAttentionMask: [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
    ],
  ),
  SingleEncodingTestCase(
    name: 'Repeated word',
    input: 'test test test test test test test test test test ',
    expectedTokens: [
      '[CLS]',
      'test',
      'test',
      'test',
      'test',
      'test',
      'test',
      'test',
      'test',
      'test',
      'test',
      '[SEP]',
    ],
    expectedIds: [
      101,
      3231,
      3231,
      3231,
      3231,
      3231,
      3231,
      3231,
      3231,
      3231,
      3231,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ),
  SingleEncodingTestCase(
    name: 'Punctuation only',
    input: '.,!?;:',
    expectedTokens: ['[CLS]', '.', ',', '!', '?', ';', ':', '[SEP]'],
    expectedIds: [101, 1012, 1010, 999, 1029, 1025, 1024, 102],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 0],
    expectedAttentionMask: [1, 1, 1, 1, 1, 1, 1, 1],
  ),
];

const _pairEncodingTestCases = [
  PairEncodingTestCase(
    name: 'Simple QA pair',
    inputA: 'What is NLP?',
    inputB: 'Natural language processing.',
    expectedTokens: [
      '[CLS]',
      'what',
      'is',
      'nl',
      '##p',
      '?',
      '[SEP]',
      'natural',
      'language',
      'processing',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      2054,
      2003,
      17953,
      2361,
      1029,
      102,
      3019,
      2653,
      6364,
      1012,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
  ),
  PairEncodingTestCase(
    name: 'Question-Answer pair',
    inputA: 'Who created BERT?',
    inputB: 'Google AI researchers.',
    expectedTokens: [
      '[CLS]',
      'who',
      'created',
      'bert',
      '?',
      '[SEP]',
      'google',
      'ai',
      'researchers',
      '.',
      '[SEP]',
    ],
    expectedIds: [
      101,
      2040,
      2580,
      14324,
      1029,
      102,
      8224,
      9932,
      6950,
      1012,
      102,
    ],
    expectedTypeIds: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
  ),
  PairEncodingTestCase(
    name: 'Short pair',
    inputA: 'Hello',
    inputB: 'World',
    expectedTokens: ['[CLS]', 'hello', '[SEP]', 'world', '[SEP]'],
    expectedIds: [101, 7592, 102, 2088, 102],
    expectedTypeIds: [0, 0, 0, 1, 1],
  ),
];

const _edgeCaseTestCases = [
  EdgeCaseTestCase(
    name: 'Empty string',
    input: '',
    expectedTokens: ['[CLS]', '[SEP]'],
    expectedIds: [101, 102],
  ),
  EdgeCaseTestCase(
    name: 'Whitespace only',
    input: '   ',
    expectedTokens: ['[CLS]', '[SEP]'],
    expectedIds: [101, 102],
  ),
  EdgeCaseTestCase(
    name: 'Tab and newline',
    input: '\t\n',
    expectedTokens: ['[CLS]', '[SEP]'],
    expectedIds: [101, 102],
  ),
  EdgeCaseTestCase(
    name: 'Single character',
    input: 'a',
    expectedTokens: ['[CLS]', 'a', '[SEP]'],
    expectedIds: [101, 1037, 102],
  ),
  EdgeCaseTestCase(
    name: 'Single punctuation',
    input: '.',
    expectedTokens: ['[CLS]', '.', '[SEP]'],
    expectedIds: [101, 1012, 102],
  ),
  EdgeCaseTestCase(
    name: 'Multiple spaces between words',
    input: 'hello    world',
    expectedTokens: ['[CLS]', 'hello', 'world', '[SEP]'],
    expectedIds: [101, 7592, 2088, 102],
  ),
  EdgeCaseTestCase(
    name: 'All caps',
    input: 'HELLO WORLD',
    expectedTokens: ['[CLS]', 'hello', 'world', '[SEP]'],
    expectedIds: [101, 7592, 2088, 102],
  ),
  EdgeCaseTestCase(
    name: 'Mixed case',
    input: 'HeLLo WoRLd',
    expectedTokens: ['[CLS]', 'hello', 'world', '[SEP]'],
    expectedIds: [101, 7592, 2088, 102],
  ),
  EdgeCaseTestCase(
    name: 'URL-like text',
    input: 'https://example.com',
    expectedTokens: [
      '[CLS]',
      'https',
      ':',
      '/',
      '/',
      'example',
      '.',
      'com',
      '[SEP]',
    ],
    expectedIds: [101, 16770, 1024, 1013, 1013, 2742, 1012, 4012, 102],
  ),
  EdgeCaseTestCase(
    name: 'Email-like text',
    input: 'test@example.com',
    expectedTokens: ['[CLS]', 'test', '@', 'example', '.', 'com', '[SEP]'],
    expectedIds: [101, 3231, 1030, 2742, 1012, 4012, 102],
  ),
  EdgeCaseTestCase(
    name: 'Currency symbols',
    input: r'$100',
    expectedTokens: ['[CLS]', r'$', '100', '[SEP]'],
    expectedIds: [101, 1002, 2531, 102],
  ),
  EdgeCaseTestCase(
    name: 'Percentage',
    input: '50%',
    expectedTokens: ['[CLS]', '50', '%', '[SEP]'],
    expectedIds: [101, 2753, 1003, 102],
  ),
];

const _offsetTestCases = [
  OffsetTestCase(
    name: 'Simple word offsets',
    input: 'hello',
    expectedOffsets: [(0, 0), (0, 5), (0, 0)],
  ),
  OffsetTestCase(
    name: 'Two words offsets',
    input: 'hello world',
    expectedOffsets: [(0, 0), (0, 5), (6, 11), (0, 0)],
  ),
  OffsetTestCase(
    name: 'Subword offsets',
    input: 'tokenization',
    expectedOffsets: [(0, 0), (0, 5), (5, 12), (0, 0)],
  ),
];
