# Contributing

## Building `backend_cpu`

Unoptimized:

```sh
cargo build
```

Optimized:

```sh
cargo build --release
```

## Building `backend_wasm`

Unoptimized:

```sh
deno run -A https://deno.land/x/wasmbuild@0.11.0/main.ts --out src/backend_wasm/lib --debug
```

Optimized:

```sh
deno run -A https://deno.land/x/wasmbuild@0.11.0/main.ts --out src/backend_wasm/lib
```
