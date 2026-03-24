/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 * @details The program checks if a string of '0's and '1's representing football
 * players contains a "dangerous" situation, defined as 7 or more consecutive
 * players of the same team.
 *
 * @author (unknown)
 *
 * @algorithm
 * This implementation uses a single pass over the string to detect a long run.
 * 1. The input string `s` is read.
 * 2. A sentinel character (' ') is appended to the string. This simplifies the loop
 *    by ensuring that any run of characters at the end of the original string is
 *    properly terminated.
 * 3. A `count` variable is initialized to 0. This variable tracks the number of
 *    consecutive identical pairs.
 * 4. The code iterates from the first character to the second-to-last character of
 *    the modified string.
 * 5. In each iteration, it compares `s[i]` with the next character `s[i+1]`.
 * 6. If they are the same, `count` is incremented. This means the run continues.
 * 7. If they are different, the run is broken, and `count` is reset to 0.
 * 8. After updating the count, it checks if `count` is 6 or more. A run of 7
 *    identical characters consists of 6 consecutive identical pairs. If this
 *    condition is met, "YES" is printed, and the program exits.
 * 9. If the loop completes without the condition being met, "NO" is printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, due to the single pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  // Append a sentinel character to handle runs at the end of the string gracefully.
  s += ' ';
  // `count` will track the number of consecutive identical pairs.
  int count = 0;

  /**
   * Block Logic: This loop iterates through the string to find consecutive characters.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == s[i + 1]) {
      // If the next character is the same, we've found another pair in the run.
      count++;
    } else {
      // If the character changes, the run is broken, so reset the counter.
      count = 0;
    }

    /**
     * Block Logic: Check for a dangerous situation.
     * A run of 7 identical characters corresponds to 6 consecutive identical pairs.
     * Therefore, if the count of pairs reaches 6, the condition is met.
     */
    if (count >= 6) {
      cout << "YES" << endl;
      return 0; // Exit early as the answer is found.
    }
  }

  // If the loop finishes, no run of 7 was found.
  cout << "NO" << endl;

  return 0;
}
