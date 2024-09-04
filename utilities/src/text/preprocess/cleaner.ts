import type { StandardizeConfig } from "../../utils/common_types.ts";

/** Simple text cleaner */
export class TextCleaner implements StandardizeConfig {
  stripHtml: boolean;
  lowercase: boolean;
  normalizeWhiteSpaces: boolean;
  stripNewlines: boolean;
  constructor({
    stripHtml = false,
    lowercase = false,
    normalizeWhiteSpaces = true,
    stripNewlines = true,
  }: StandardizeConfig = {}) {
    this.stripHtml = stripHtml;
    this.lowercase = lowercase;
    this.normalizeWhiteSpaces = normalizeWhiteSpaces;
    this.stripNewlines = stripNewlines;
  }
  clean(text: string): string;
  clean(text: string[]): string[];
  clean(text: string | string[]) {
    if (Array.isArray(text)) {
      return text.map((line) => preprocess(line, this));
    }
    return preprocess(text, this);
  }
}

/** Function for quick cleaning of text */
export function preprocess(
  text: string,
  {
    stripHtml = false,
    lowercase = false,
    normalizeWhiteSpaces = true,
    stripNewlines = true,
  }: StandardizeConfig = {},
): string {
  if (lowercase) {
    text = text.toLowerCase();
  }
  if (stripHtml) {
    text = text.replace(/<([^>]+)>/g, " ");
  }
  if (stripNewlines) {
    text = text.replace(/\n/g, " ");
  }
  if (normalizeWhiteSpaces) {
    text = text.replace(/\s\s+/g, " ");
  }
  return text;
}
