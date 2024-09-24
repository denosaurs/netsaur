/** Post-processing step only occuring during prediction routine */
export type PostProcessor =
  | { type: "none" }
  | { type: "sign" }
  | { type: "step"; config: StepFunctionConfig };

type StepFunctionConfig = { thresholds: number[]; values: number[] };

export function PostProcess(pType: "none" | "sign"): PostProcessor;
export function PostProcess(
  pType: "step",
  config: StepFunctionConfig
): PostProcessor;
export function PostProcess(
  pType: "none" | "sign" | "step",
  config?: StepFunctionConfig
) {
  if (pType === "none" || pType === "sign") {
    return { type: pType };
  }
  return { type: pType, config };
}
