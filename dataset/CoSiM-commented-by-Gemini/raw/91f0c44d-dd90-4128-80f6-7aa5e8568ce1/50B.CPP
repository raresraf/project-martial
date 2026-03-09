/**
 * @file 50B.CPP
 * @brief Calculates the sum of squares of character frequencies in a given string.
 *
 * This program reads a line of text, counts the occurrences of each character
 * (assuming an 8-bit character set), and then computes the sum of the squares
 * of these counts. This is equivalent to counting the number of ordered pairs
 * of indices (i, j) such that the characters at these indices are the same.
 *
 * @algorithm Character Frequency Counting
 * @complexity
 * - Time: O(L) where L is the length of the input string. This is because the
 *   program iterates through the string once to count characters and then
 *   iterates through a fixed-size array (256 elements).
 * - Space: O(1) as the space used by the `counts` vector is constant (256 elements),
 *   not dependent on the input string length.
 */

#include <cstdio>
#include <iostream>
#include <vector>

int main() {

  /**
   * @brief Defines the size of the character set.
   * Functional Utility: Assumes an 8-bit ASCII or extended ASCII character set,
   * allowing for direct mapping of character codes to array indices.
   */
  const int N = 256;
  std::vector<long long> counts(N, 0);

  /**
   * @brief Block Logic: Reads a line of input from stdin.
   */
  std::string input;
  getline(std::cin, input);
  
  /**
   * @brief Block Logic: Counts the frequency of each character in the input string.
   * Invariant: At the end of this loop, `counts[c]` will hold the total number
   * of occurrences of character `c` in the input string.
   */
  for (long k = 0; k < input.size(); k++) {
    ++counts[input[k]];
  }

  long long total(0);
  /**
   * @brief Block Logic: Calculates the sum of the squares of character counts.
   * Logic: For each character `c` with frequency `f`, this adds `f*f` to the total.
   * This corresponds to the number of ways to choose two characters of type `c` from
   * the string (with replacement, order matters).
   */
  for (long k = 0; k < N; k++) {
    total += counts[k] * counts[k];
  }

  /**
   * @brief Output the final calculated sum.
   */
  printf("%lld\n", total);

  return 0;
}