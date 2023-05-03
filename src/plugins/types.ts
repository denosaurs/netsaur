/**
 * Netsaur Plugin Interface
 */
export interface Plugin {
  /**
   * Plugin name.
   */
  name: string;

  /**
   * Plugin version.
   */
  version: `${number}.${number}.${number}` | `v${number}.${number}.${number}`;
  // deno-lint-ignore no-explicit-any
  [method: string]: any;
}
