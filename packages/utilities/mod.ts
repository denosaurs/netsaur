/**
 * Machine Learning utilities for TypeScript.
 *
 * @example
 * ```ts
 * const data = [
 *   "twinkle twinkle little star",
 *   "How I wonder what you are",
 *   "up above the world so high",
 *   "like a diamond in the sky",
 * ];
 *
 * // Clean the text
 * const cleaner = new TextCleaner({
 *   lowercase: true,
 *   stripHtml: true,
 *   stripNewlines: true,
 *   normalizeWhiteSpaces: true,
 * });
 * x = cleaner.clean(x);
 *
 * // Tokenize the text
 * const tokenizer = new SplitTokenizer();
 * tokenizer.fit(x);
 * const x_tokens = tokenizer.transform(x);
 *
 * // Vectorize the tokens
 * const vectorizer = new CountVectorizer(tokenizer.vocabulary.size);
 * const x_vec = vectorizer.transform(x_tokens, "f32");
 *
 * // Apply Tf-Idf transformation
 * const transformer = new TfIdfTransformer();
 * console.log(transformer.fit(x_vec).transform(x_vec));
 * ```
 * @module
 */
export * from "./src/mod.ts";
