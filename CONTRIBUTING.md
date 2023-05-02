# Contributing

### Building `backend_cpu`

Unoptimized: 

```
cargo build
```

Optimized: 

```
cargo build --release
```

### Building `backend_wasm`

Unoptimized: 

```
deno run -A https://deno.land/x/wasmbuild@0.11.0/main.ts --out src/backend_wasm/lib --debug
```

Optimized: 

```
deno run -A https://deno.land/x/wasmbuild@0.11.0/main.ts --out src/backend_wasm/lib
```