# Contributing

## Building `backends/cpu`

Unoptimized:

```sh
cargo build
```

Optimized:

```sh
cargo build --release
```

## Building `backends/wasm`

Unoptimized:

```sh
deno run -A https://deno.land/x/wasmbuild@0.15.1/main.ts --out src/backends/wasm/lib --debug
```

Optimized:

```sh
deno run -A https://deno.land/x/wasmbuild@0.15.1/main.ts --out src/backends/wasm/lib
```
