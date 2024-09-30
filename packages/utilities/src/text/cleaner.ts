import { DefaultIgnoreList } from "../constants/stop_words.ts";

interface StandardizeConfig {
  /** Whether to convert everything to lowercase before fitting / transforming */
  lowercase?: boolean;
  /** Whether to strip HTML tags */
  stripHtml?: boolean;
  /** Whether to replace multiple whitespaces. */
  normalizeWhiteSpaces?: boolean;
  /** Strip Newlines */
  stripNewlines?: boolean;
  removeMentions?: boolean;
  keepOnlyAlphaNumeric?: boolean;
  /** Remove stop words from text */
  removeStopWords?: "english" | false | string[];
}

/** Simple text cleaner */
export class TextCleaner implements StandardizeConfig {
  stripHtml: boolean;
  lowercase: boolean;
  normalizeWhiteSpaces: boolean;
  stripNewlines: boolean;
  removeStopWords: false | "english" | string[];
  removeMentions: boolean;
  keepOnlyAlphaNumeric: boolean;
  constructor({
    stripHtml = false,
    lowercase = false,
    normalizeWhiteSpaces = true,
    stripNewlines = true,
    removeStopWords = false,
    removeMentions = false,
    keepOnlyAlphaNumeric = false,
  }: StandardizeConfig = {}) {
    this.stripHtml = stripHtml;
    this.lowercase = lowercase;
    this.normalizeWhiteSpaces = normalizeWhiteSpaces;
    this.stripNewlines = stripNewlines;
    this.removeStopWords = removeStopWords;
    this.keepOnlyAlphaNumeric = keepOnlyAlphaNumeric;
    this.removeMentions = removeMentions;
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
    removeStopWords = false,
    removeMentions = false,
    keepOnlyAlphaNumeric = false,
  }: StandardizeConfig = {}
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
  if (removeStopWords) {
    const stopWords =
      removeStopWords === "english" ? DefaultIgnoreList : removeStopWords;
    text = text
      .split(" ")
      .filter((x) => !stopWords.includes(x))
      .join(" ");
  }
  if (removeMentions) {
    text = text.replace(/@\w+/g, "")
  }
  if (keepOnlyAlphaNumeric) {
    text = text.replace(/[^a-zA-Z0-9 ]+/g, "")
  }
  return text;
}
