# Dart Coding Style Guide

Coding styles and design principles based on Effective Dart.

## 1. Naming Conventions

### Identifier Notation

| Identifier Type | Notation | Example |
|-------------|--------|------|
| Type (Classes, Enums, Typedefs) | `UpperCamelCase` | `SliderMenu`, `HttpRequest` |
| Type Parameter | `UpperCamelCase` | `T`, `E`, `Future<String>` |
| Extension | `UpperCamelCase` | `extension MyFancyList on List` |
| Variable, Parameter, Member | `lowerCamelCase` | `itemCount`, `httpRequest` |
| **const constant** | `lowerCamelCase` | `defaultTimeout` (SCREAMING_CAPS ❌) |
| Package, Library, File | `lowercase_with_underscores` | `http_connection.dart` |
| Import Prefix | `lowercase_with_underscores` | `import ... as dart_math` |

### Abbreviation Handling

| Abbreviation Length | Notation | Example |
|-----------|--------|------|
| More than 2 characters | Treat as a word | `HttpConnection` ✅, `HTTPConnection` ❌ |
| 2 characters | All uppercase | `IOStream`, `UIHandler` |

### Meaning of Underscore (`_`)

```dart
// ✅ Library private (syntactic meaning)
class _InternalHelper {}

// ❌ Unnecessary for local variables
void process() {
  var _value = 10;  // No meaning
}

// ✅ Unused callback parameter
list.forEach((_, index) => print(index));
```

## 2. Code Structuring

### Import Order

```dart
// 1. Dart SDK libraries
import 'dart:async';
import 'dart:io';

// 2. External packages
import 'package:flutter/material.dart';
import 'package:http/http.dart';

// 3. Relative path imports
import 'src/utils.dart';

// export comes after import, in a separate section
export 'src/public_api.dart';
```

- Sort alphabetically within each section
- Dependency scope: Global → Local

### Formatting

```bash
dart format .  # Use automatic formatting
```

- Line length: 80 characters
- Follow `dart format` results

## 3. Null Safety

### Prohibit Unnecessary Null Initialization

```dart
// ❌ Redundant
String? name = null;

// ✅ Automatically null
String? name;
```

### Caution when using late

```dart
// ❌ Cannot check if initialized
late String value;
if (/* is value initialized? */) {}  // No API

// ✅ If check is needed, use nullable
String? value;
if (value != null) {
  // Use safely
}
```

### Utilize Type Promotion

```dart
String? maybeText;

// ✅ Automatic promotion through flow analysis
if (maybeText != null) {
  print(maybeText.length);  // Promoted to String
}

// Fields cannot be promoted → Copy to local variable
class Example {
  String? field;

  void process() {
    final localField = field;  // Copy
    if (localField != null) {
      print(localField.length);  // Promoted
    }
  }
}
```

## 4. Collection Usage

### Use Literals

```dart
// ❌
var list = List<int>();
list.add(1);

// ✅ Collection literals + collection if/for
var list = [
  1,
  2,
  if (condition) 3,
  for (var i in items) i * 2,
];
```

### Checking if Empty

```dart
// ❌ length can be O(N)
if (list.length == 0) {}

// ✅ isEmpty is O(1)
if (list.isEmpty) {}
if (list.isNotEmpty) {}
```

### Type Conversion

```dart
// ❌ List.from performs runtime type check
var copy = List<int>.from(original);

// ✅ toList preserves types and is optimized
var copy = original.toList();

// ❌ cast is a lazy wrapper (type check on every access)
var filtered = list.where((x) => x is int).cast<int>();

// ✅ whereType is efficient
var filtered = list.whereType<int>();
```

### Loops

```dart
// ❌ forEach does not support break, return, await
list.forEach((item) {
  // cannot use await
});

// ✅ for-in recommended
for (var item in list) {
  await process(item);
  if (done) break;
}
```

## 5. Function and Member Design

### Use Tear-off

```dart
// ❌ Unnecessary lambda wrapping
names.forEach((name) { print(name); });

// ✅ Pass function reference directly
names.forEach(print);
```

### Do Not Store Computable Values

```dart
// ❌ Risk of state inconsistency
class Rectangle {
  double width, height;
  double area;  // Caching

  void updateWidth(double w) {
    width = w;
    area = width * height;  // Requires synchronization
  }
}

// ✅ Real-time calculation via getter
class Rectangle {
  double width, height;

  double get area => width * height;
}
```

### Field vs Getter/Setter

```dart
// ❌ Do not wrap in advance
class Point {
  int _x;
  int get x => _x;
  set x(int value) => _x = value;
}

// ✅ API compatible even if converted later
class Point {
  int x;  // Can be changed to getter later
}
```

## 6. Parameter Design

### Booleans as Named Parameters

```dart
// ❌ Ambiguous meaning
doSomething(true);

// ✅ Clear meaning
doSomething(enableLogging: true);
```

### Avoid Required Null Parameters

```dart
// ❌
void process({required String? name}) {
  // Forces passing null
}

// ✅
void process({String? name}) {
  // Can be omitted
}

// Or provide a default value
void process({String name = 'default'}) {}
```

## 7. Class Design

### Class Modifiers (Dart 3.0+)

```dart
// Inheritance prohibited
final class ImmutablePoint {
  final int x, y;
}

// Use only as an interface
interface class Drawable {
  void draw();
}

// Mixin only
mixin Loggable {
  void log(String message) => print(message);
}

// Mixin + Class
mixin class Counter {
  int count = 0;
  void increment() => count++;
}
```

### Implementing Equality

```dart
class Point {
  final int x, y;

  // hashCode is essential when overriding ==
  @override
  bool operator ==(Object other) =>
      other is Point && other.x == x && other.y == y;

  @override
  int get hashCode => Object.hash(x, y);
}
```

**Note**: Avoid defining equality for mutable objects (problematic for hash-based collections)

## 8. Asynchronous Processing

### Return Type

```dart
// ❌ void return → cannot wait for completion or handle errors
void fetchData() async {
  await api.fetch();
}

// ✅ Return Future<void>
Future<void> fetchData() async {
  await api.fetch();
}
```

### Avoid FutureOr

```dart
// ❌ Caller must check type every time
FutureOr<String> getValue() { ... }

// ✅ Always return Future
Future<String> getValue() async { ... }
// OR
Future<String> getValue() => Future.value(cachedValue);
```

## 9. Error Handling

### Error vs Exception

| Type | Purpose | Handling |
|------|------|------|
| `Error` | Program bug (ArgumentError, IndexError) | Prohibit catch, requires fix |
| `Exception` | Runtime exception (IOException) | Recover via catch |

### Error Propagation

```dart
// ❌ Loss of stack trace
try {
  risky();
} catch (e) {
  throw e;
}

// ✅ Preserve original stack trace
try {
  risky();
} catch (e) {
  rethrow;
}

// ❌ Swallowing all errors
try {
  risky();
} catch (e) {
  // Hides even bugs
}

// ✅ Handle only specific exceptions
try {
  risky();
} on IOException catch (e) {
  handleIO(e);
}
```

## 10. Documentation

### Writing Doc Comments

```dart
/// Fetches user information from the server.
///
/// [userId] is the unique ID of the user to look up.
/// Returns `null` if the user is not found.
///
/// ```dart
/// final user = await fetchUser('123');
/// ```
Future<User?> fetchUser(String userId) async { ... }
```

- Use `///` (dart doc compatible)
- Reference identifiers with square brackets `[parameter]`
- The first sentence should be a complete sentence

## Summary Table

### Collection Best Practices

| Task | Avoid | Recommended |
|------|------|------|
| Creation | `List(); l.add(1);` | `[1]` |
| Checking Empty | `list.length == 0` | `list.isEmpty` |
| Type Filtering | `.where().cast<T>()` | `.whereType<T>()` |
| Copying | `List.from(list)` | `list.toList()` |
| Iteration | `forEach((x) {...})` | `for (var x in list)` |

### Key Design Principles

| Category | Rule |
|------|------|
| Null Safety | No explicit null initialization, avoid checking late initialization |
| Members | Do not wrap fields in getters/setters in advance |
| Parameters | No positional boolean arguments |
| Equality | `hashCode` is required when overriding `==` |
| Async | Asynchronous functions return `Future<void>` |
| Errors | No catching all errors without an `on` clause |