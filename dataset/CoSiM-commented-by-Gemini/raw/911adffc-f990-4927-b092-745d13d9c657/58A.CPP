/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" subsequence problem (e.g., Codeforces 58A).
 * @details This program determines if the word "hello" is a subsequence of a given input
 * string. It uses a clean and efficient two-pointer approach.
 *
 * @algorithm Two-Pointer Subsequence Check
 * @complexity
 * - Time: O(L), where L is the length of the input string, due to a single pass.
 * - Space: O(L) to store the input string.
 */
#include <iostream>
#include <string> // It's good practice to include necessary headers explicitly.
using namespace std;

int main() {
  string s; // The input string from the user.
  cin >> s;

  // 'i' is the pointer for the input string `s`.
  // 'j' is the pointer for the target string `g`.
  unsigned int i = 0, j = 0;

  string g = "hello";      // The target subsequence.
  string output = "NO"; // The default result.

  /**
   * @block Subsequence Search
   * @logic Iterates through the input string `s` while trying to match characters
   * of the target string `g` in order.
   */
  while (i < s.size()) {
    // Check if the current character in `s` matches the current target character in `g`.
    // The post-increment `i++` ensures we always advance through the input string.
    if (s[i++] == g[j]) {
      // If a match is found, advance the target pointer `j` to look for the next character.
      j++;
    }

    // If `j` has reached the length of "hello", all characters have been found in order.
    if (j >= g.size()) {
      output = "YES";
      break; // Exit the loop as soon as the subsequence is confirmed.
    }
  }

  cout << output << endl;

  return 0;
  // The second `return 0;` is redundant and unreachable.
}
