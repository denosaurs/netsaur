# Contributing

## Building `backends/cpu`

Unoptimized:

```sh
cargo build --release -p netsaur
```

Optimized:

```sh
deno run build:cpu
```

## Building `backends/wasm`

Unoptimized:

```sh
deno -Ar jsr:@deno/wasmbuild@0.17.2 -p netsaur --out src/backends/wasm/lib --debug
```

Optimized:

```sh
deno run build:wasm
```

## Building `tokenizers`

Unoptimized:

```sh
deno -Ar jsr:@deno/wasmbuild@0.17.2 -p netsaur-tokenizers --out tokenizers/lib --debug
```

Optimized:

```sh
deno run build:tokenizers
```

## Building everything

Optimized:

```sh
deno run build
```
