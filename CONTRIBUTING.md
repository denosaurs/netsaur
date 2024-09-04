# Contributing

## Building `backends/cpu`

Unoptimized:

```sh
cargo build --release -p netsaur
```

Optimized:

```sh
deno task build:cpu
```

## Building `backends/wasm`

Unoptimized:

```sh
deno run -Ar jsr:@deno/wasmbuild@0.17.2 -p netsaur --out src/backends/wasm/lib --debug
```

Optimized:

```sh
deno task build:wasm
```

## Building `tokenizers`

Unoptimized:

```sh
deno run -Ar jsr:@deno/wasmbuild@0.17.2 -p netsaur-tokenizers --out tokenizers/lib --debug
```

Optimized:

```sh
deno task build:tokenizers
```

## Building everything

Optimized:

```sh
deno task build
```
