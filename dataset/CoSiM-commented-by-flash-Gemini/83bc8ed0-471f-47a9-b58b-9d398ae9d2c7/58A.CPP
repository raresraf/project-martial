/**
 * @file 58A.CPP
 * @brief Solution to a variation of the "Chat room" competitive programming problem.
 *
 * This program checks if a given input string `s` contains the word "hello" as a subsequence,
 * with a specific constraint: it requires at least two 'l' characters to appear
 * in sequence after 'h' and 'e' have been found, followed by an 'o'.
 *
 * Algorithm: Sequential subsequence matching with character count validation.
 * Time Complexity: O(L_s) where L_s is the length of the input string `s`, as it
 *                  involves a single pass over the string.
 * Space Complexity: O(1) as it uses a fixed number of boolean flags and integer counters.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return int Returns 0 on successful execution.
 */
int main() {
  string s;
  /**
   * @brief Reads the input string from standard input.
   *
   * Pre-condition: A string is available on standard input.
   * Post-condition: `s` stores the input string.
   */
  cin >> s;

  // Boolean flags to track if 'h', 'e', 'l', 'o' have been encountered in the correct order.
  bool hh = false, ee = false, ll = false, oo = false;
  // Counters to track the number of occurrences of each character, used for 'l' count specifically.
  int h = 0, e = 0, l = 0, o = 0;

  /**
   * @brief Iterates through the input string `s` to find "hello" as a subsequence
   *        with the specific "two 'l's" requirement.
   *
   * Invariant: `hh`, `ee`, `ll`, `oo` flags are updated based on sequential character matches.
   *            `h`, `e`, `l`, `o` counters track character occurrences.
   * Pre-condition: `s` is initialized with the input string.
   * Post-condition: Flags and counters reflect the matching status of "hello" characters.
   */
  for (int i = 0; i < s.length(); i++) {
    // Conditional Logic: Matches 'h'.
    if (s[i] == 'h') {
      h++;
      hh = true;
    }
    // Conditional Logic: Matches 'e' only if 'h' has already been found.
    if (s[i] == 'e' && hh) {
      ee = true;
      e++;
    }
    // Conditional Logic: Matches 'l' only if 'h' and 'e' have already been found.
    if (s[i] == 'l' && hh && ee) {
      ll = true;
      l++;
    }
    // Conditional Logic: Matches 'o' only if 'h', 'e', and 'l' (at least one) have already been found.
    if (s[i] == 'o' && hh && ee && ll) {
      oo = true;
      o++;
    }
  }

  /**
   * @brief Determines and prints the final result based on the matching flags and 'l' count.
   *
   * Conditional Logic: Checks if all characters ('h', 'e', at least two 'l's, 'o')
   *                    were found in the correct sequence.
   * Functional Utility: Outputs "YES" if all conditions are met, otherwise "NO".
   */
  if (hh && ee && ll && oo) {
    // Conditional Logic: Final check to ensure at least two 'l's were found.
    if (l >= 2)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  } else
    cout << "NO" << endl;
  return 0;
}