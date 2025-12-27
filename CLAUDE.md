# Project Rules

## Version Update Checklist

When updating the package version, you MUST update the following files:

1. **pubspec.yaml** - Update the `version` field
2. **README.md** - Update version references if any (e.g., installation instructions)
3. **CHANGELOG.md** - Add a new entry describing the changes in the new version

### CHANGELOG.md Format

Follow the Keep a Changelog format:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features
```