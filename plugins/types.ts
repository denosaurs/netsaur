export interface Plugin {
    name: string;
    version: `${number}.${number}.${number}` | `v${number}.${number}.${number}`;
    // deno-lint-ignore no-explicit-any
    [method: string]: any
}