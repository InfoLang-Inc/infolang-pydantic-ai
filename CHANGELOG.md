# Changelog

All notable changes to `infolang-pydantic-ai` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-07-15

### Added
- Initial release: InfoLang semantic-memory tools for [Pydantic AI](https://ai.pydantic.dev).
- `InfoLangDeps` dependency object (wraps `infolang.AsyncInfoLang`) for
  `Agent(deps_type=InfoLangDeps)` injection, with `from_api_key` / `from_client`
  constructors and async context-manager lifecycle.
- Four context tools — `recall`, `investigate`, `remember`, `forget` — usable
  individually (`Agent(tools=[...])`) or bundled via `infolang_toolset()`.
- Typed `RecallHit` / `RecallResponse` result models (schema-first return types).
- Offline unit tests mock the HTTP layer with `respx`; agent-integration tests
  drive the tools through Pydantic AI's `TestModel`. Optional live smoke test
  gated on `INFOLANG_API_KEY`.
