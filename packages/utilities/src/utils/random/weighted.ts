// Check out https://github.com/retraigo/fortuna

export interface WeightedChoice<ItemType> {
  result: ItemType;
  chance: number;
}

/**
 * Roll one from an array of weighted choices.
 * @param {WeightedChoice[]} choices - Choices to roll from.
 * @param {number} totalChance - Sum of all chance properties.
 * @returns {WeightedChoice} Item rolled.
 */

export function useWeighted<ItemType>(
  choices: WeightedChoice<ItemType>[],
): WeightedChoice<ItemType> {
  const total = choices.reduce(
    (acc: number, val: WeightedChoice<ItemType>) => acc + val.chance,
    0,
  );
  const result = Math.random() * total;
  let going = 0.0;
  for (let i = 0; i < choices.length; ++i) {
    going += choices[i].chance;
    if (result < going) return choices[i];
  }
  return choices[Math.floor(Math.random() * choices.length)];
}
