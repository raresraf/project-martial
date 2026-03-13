/**
 * @file 339A.CPP
 * @brief Solution for CodeForces Problem 339A - "Helpful Maths".
 *
 * @details
 * The program takes a string representing a sum of the numbers 1, 2, and 3
 * (e.g., "3+2+1") and rearranges the numbers in non-decreasing order
 * (e.g., "1+2+3").
 *
 * @algorithm Counting Sort
 * 1. Count the occurrences of the numbers '1', '2', and '3' in the input string.
 *    The numbers are always separated by a '+' sign, so we only need to inspect
 *    characters at even indices.
 * 2. Reconstruct the string:
 *    a. Print the first number without a leading '+'.
 *    b. Print the remaining numbers, each preceded by a '+'.
 *
 * @time_complexity O(L), where L is the length of the string, as we iterate through it once to count and once to print.
 * @space_complexity O(1), as the count array size is fixed.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n[3] = {0}; // n[0] counts '1's, n[1] counts '2's, n[2] counts '3's.
  int i = 0;
  cin >> s;

  /**
   * @brief Count the occurrences of each number (1, 2, 3).
   * The loop iterates through the string, stepping by 2 to only look at the
   * characters representing numbers, skipping the '+' signs.
   */
  while (i < s.length()) {
    // `s[i] - '1'` cleverly converts character '1','2','3' to index 0,1,2.
    n[s[i] - '1']++;
    i += 2;
  }

  /**
   * @brief Print the first term of the sum.
   * This loop finds the smallest number that appeared in the input and prints it.
   * The `n[i]--` is a post-decrement; it uses the value of n[i] to check the condition
   * and then decrements it. This ensures this number isn't printed again here.
   */
  for (i = 0; i < 3; i++) {
    if (n[i]--) {
      cout << i + 1;
      break; // Exit after printing the very first term.
    }
  }

  /**
   * @brief Print the rest of the terms, separated by '+'.
   * This continues from where the previous loop left off (`i`).
   * The `while(n[i]--)` loop will run `n[i]` times (using the decremented values),
   * printing each term preceded by a '+'.
   */
  for (; i < 3; i++) {
    while (n[i]--)
      cout << "+" << i + 1;
  }

  return 0;
}
