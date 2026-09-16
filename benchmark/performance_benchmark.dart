import 'dart:io';
import 'dart:typed_data';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() async {
  print('=' * 70);
  print('dart_bert_tokenizer Performance Benchmark');
  print('=' * 70);
  print('');

  // Load vocabulary
  final vocabPath = _findVocabFile();
  if (vocabPath == null) {
    print('ERROR: vocab.txt not found. Using demo vocabulary.');
    await _runWithDemoVocab();
    return;
  }

  print('Loading vocabulary from: $vocabPath');
  final sw = Stopwatch()..start();
  final tokenizer = WordPieceTokenizer.fromVocabFileSync(vocabPath);
  sw.stop();
  print(
    'Vocabulary loaded: ${tokenizer.vocab.size} tokens in ${sw.elapsedMilliseconds}ms',
  );
  print('');

  // Run benchmarks
  await _runSingleEncodingBenchmark(tokenizer);
  await _runBatchEncodingBenchmark(tokenizer);
  await _runParallelBenchmark(tokenizer);
  await _runThroughputBenchmark(tokenizer);
  await _runMemoryBenchmark(tokenizer);

  print('=' * 70);
  print('Benchmark Complete');
  print('=' * 70);
}

Future<void> _runWithDemoVocab() async {
  final vocab = Vocabulary.fromTokens([
    '[PAD]',
    ...List.generate(99, (i) => '[unused$i]'),
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '[MASK]',
    ...List.generate(896, (i) => '[unused${99 + i}]'),
    'the',
    'a',
    'is',
    'it',
    'this',
    'that',
    'what',
    'how',
    'to',
    'of',
    'and',
    'hello',
    'world',
    'test',
    'word',
    'sentence',
    'text',
    'quick',
    'brown',
    'fox',
    'jumps',
    'over',
    'lazy',
    'dog',
    'machine',
    'learning',
    'deep',
    'neural',
    'network',
    'transformer',
    'bert',
    'token',
    'embedding',
    ',',
    '.',
    '!',
    '?',
    "'",
    '##s',
    '##ed',
    '##ing',
    '##er',
    '##est',
    '##ly',
    '##tion',
    '##ment',
  ]);
  final tokenizer = WordPieceTokenizer(vocab: vocab);

  await _runSingleEncodingBenchmark(tokenizer);
  await _runBatchEncodingBenchmark(tokenizer);
  await _runParallelBenchmark(tokenizer);
  await _runThroughputBenchmark(tokenizer);
  await _runMemoryBenchmark(tokenizer);
}

String? _findVocabFile() {
  final paths = ['vocab.txt', 'benchmark/vocab.txt', '../vocab.txt'];
  for (final path in paths) {
    if (File(path).existsSync()) return path;
  }
  return null;
}

Future<void> _runSingleEncodingBenchmark(WordPieceTokenizer tokenizer) async {
  print('-' * 70);
  print('1. SINGLE ENCODING LATENCY');
  print('-' * 70);

  final testCases = [
    ('Short', 'Hello, world!'),
    ('Medium', 'The quick brown fox jumps over the lazy dog.'),
    (
      'Long',
      'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with many layers.',
    ),
  ];

  for (final (name, text) in testCases) {
    // Warmup
    for (var i = 0; i < 100; i++) {
      tokenizer.encode(text);
    }

    // Benchmark
    const iterations = 1000;
    final sw = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      tokenizer.encode(text);
    }
    sw.stop();

    final encoding = tokenizer.encode(text);
    final avgUs = (sw.elapsedMicroseconds / iterations).toStringAsFixed(1);
    print('  $name (${encoding.length} tokens): ${avgUs}μs/encode');
  }
  print('');
}

Future<void> _runBatchEncodingBenchmark(WordPieceTokenizer tokenizer) async {
  print('-' * 70);
  print('2. BATCH ENCODING (Sequential)');
  print('-' * 70);

  final texts = List.generate(
    100,
    (i) => 'This is test sentence number $i for batch encoding benchmark.',
  );

  // Warmup
  tokenizer.encodeBatch(texts.take(10).toList());

  // Benchmark different batch sizes
  for (final batchSize in [10, 50, 100]) {
    final batch = texts.take(batchSize).toList();

    final sw = Stopwatch()..start();
    final results = tokenizer.encodeBatch(batch);
    sw.stop();

    final totalTokens = results.fold<int>(0, (sum, e) => sum + e.length);
    final tokensPerMs = (totalTokens / sw.elapsedMilliseconds).toStringAsFixed(
      0,
    );
    print(
      '  Batch $batchSize: ${sw.elapsedMilliseconds}ms ($tokensPerMs tokens/ms)',
    );
  }
  print('');
}

Future<void> _runParallelBenchmark(WordPieceTokenizer tokenizer) async {
  print('-' * 70);
  print('3. PARALLEL vs SEQUENTIAL BATCH ENCODING');
  print('-' * 70);

  final texts = List.generate(
    100,
    (i) =>
        'This is test sentence number $i for parallel encoding benchmark. '
        'Machine learning and deep learning are transforming many industries.',
  );

  // Warmup
  tokenizer.encodeBatch(texts.take(10).toList());
  await tokenizer.encodeBatchParallel(texts.take(10).toList());

  for (final batchSize in [16, 32, 64, 100]) {
    final batch = texts.take(batchSize).toList();

    // Sequential
    final swSeq = Stopwatch()..start();
    final seqResults = tokenizer.encodeBatch(batch);
    swSeq.stop();

    // Parallel
    final swPar = Stopwatch()..start();
    final parResults = await tokenizer.encodeBatchParallel(batch);
    swPar.stop();

    final speedup = (swSeq.elapsedMilliseconds / swPar.elapsedMilliseconds)
        .toStringAsFixed(2);
    final seqMs = swSeq.elapsedMilliseconds;
    final parMs = swPar.elapsedMilliseconds;

    print('  Batch $batchSize:');
    print('    Sequential: ${seqMs}ms');
    print('    Parallel:   ${parMs}ms (${speedup}x speedup)');

    // Verify results match
    var match = true;
    for (var i = 0; i < seqResults.length; i++) {
      if (seqResults[i].ids.length != parResults[i].ids.length) {
        match = false;
        break;
      }
    }
    if (!match) print('    ⚠️ Results mismatch!');
  }
  print('');
}

Future<void> _runThroughputBenchmark(WordPieceTokenizer tokenizer) async {
  print('-' * 70);
  print('4. THROUGHPUT (tokens/second)');
  print('-' * 70);

  // Generate diverse test texts
  final texts = <String>[];
  final baseTexts = [
    'Hello world!',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is transforming industries.',
    'Natural language processing enables computers to understand human language.',
    'Transformers have revolutionized the field of deep learning and NLP.',
  ];

  for (var i = 0; i < 200; i++) {
    texts.add(baseTexts[i % baseTexts.length]);
  }

  // Warmup
  tokenizer.encodeBatch(texts.take(50).toList());

  // Benchmark
  final sw = Stopwatch()..start();
  final results = tokenizer.encodeBatch(texts);
  sw.stop();

  final totalTokens = results.fold<int>(0, (sum, e) => sum + e.length);
  final tokensPerSec = (totalTokens / sw.elapsedMilliseconds * 1000).round();

  print('  Texts processed: ${texts.length}');
  print('  Total tokens: $totalTokens');
  print('  Time: ${sw.elapsedMilliseconds}ms');
  print('  Throughput: ${_formatNumber(tokensPerSec)} tokens/sec');
  print('');
}

Future<void> _runMemoryBenchmark(WordPieceTokenizer tokenizer) async {
  print('-' * 70);
  print('5. MEMORY EFFICIENCY (Typed Arrays)');
  print('-' * 70);

  final text = 'This is a test sentence for memory benchmark with many tokens.';
  final encoding = tokenizer.encode(text);

  print('  Encoding length: ${encoding.length} tokens');
  print('');
  print('  Field types (P1 optimization):');
  print('    ids:              ${encoding.ids.runtimeType}');
  print('    typeIds:          ${encoding.typeIds.runtimeType}');
  print('    attentionMask:    ${encoding.attentionMask.runtimeType}');
  print('    specialTokensMask: ${encoding.specialTokensMask.runtimeType}');
  print('');

  // Calculate memory savings
  final numTokens = encoding.length;

  // Before optimization: List<int> uses ~8 bytes per element (boxed int)
  final beforeIds = numTokens * 8;
  final beforeTypeIds = numTokens * 8;
  final beforeAttention = numTokens * 8;
  final beforeSpecial = numTokens * 8;
  final totalBefore =
      beforeIds + beforeTypeIds + beforeAttention + beforeSpecial;

  // After optimization: Int32List (4 bytes) and Uint8List (1 byte)
  final afterIds = numTokens * 4; // Int32List
  final afterTypeIds = numTokens * 1; // Uint8List
  final afterAttention = numTokens * 1; // Uint8List
  final afterSpecial = numTokens * 1; // Uint8List
  final totalAfter = afterIds + afterTypeIds + afterAttention + afterSpecial;

  final savings = ((1 - totalAfter / totalBefore) * 100).toStringAsFixed(1);

  print('  Memory comparison (numeric fields, $numTokens tokens):');
  print('    Before (List<int>):  $totalBefore bytes');
  print('    After (typed arrays): $totalAfter bytes');
  print('    Savings: $savings%');
  print('');

  // Verify typed array types
  final isOptimized =
      encoding.ids is Int32List &&
      encoding.typeIds is Uint8List &&
      encoding.attentionMask is Uint8List &&
      encoding.specialTokensMask is Uint8List;

  print('  Typed arrays verified: ${isOptimized ? "✅ Yes" : "❌ No"}');
  print('');
}

String _formatNumber(int n) {
  if (n >= 1000000) {
    return '${(n / 1000000).toStringAsFixed(2)}M';
  } else if (n >= 1000) {
    return '${(n / 1000).toStringAsFixed(0)}K';
  }
  return n.toString();
}
