/**
 * @file 71A.CPP
 * @brief "Way Too Long Words" - A string abbreviation problem.
 *
 * This program reads an integer `t`, indicating the number of test cases. For each
 * test case, it reads a word. If the word's length is greater than 10 characters,
 * it prints a special abbreviation: the first character, followed by the number of
 * characters between the first and the last, and finally the last character.
 * Otherwise, it prints the word as is.
 *
 * @algorithm
 * 1. Read the number of test cases, `t`.
 * 2. Loop `t` times:
 *    a. Read the input string, `s`.
 *    b. Check if the length of `s` is greater than 10.
 *    c. If it is, construct the abbreviation: `s[0]` (first character), `s.size() - 2`
 *       (the count of characters between the first and last), and `s.back()` (last character).
 *       Print this abbreviation.
 *    d. If the length is not greater than 10, print the string `s` unchanged.
 *
 * @complexity
 * Time Complexity: O(t * L), where `t` is the number of test cases and `L` is the maximum
 * length of a string. The string operations (accessing characters and getting size) are O(1).
 * Space Complexity: O(L), to store the longest input string.
 */

// Includes all standard C++ libraries for convenience, a common practice in competitive programming.
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `t` stores the number of test cases.
  int t;
  // `s` will store each word to be processed.
  string s;

  // Read the total number of test cases.
  cin >> t;

  /**
   * @brief Main loop to process each test case.
   * The loop will run `t` times.
   */
  while (t--) {
    // Read the word for the current test case.
    cin >> s;

    // Pre-condition: Check if the word is "too long".
    if (s.size() > 10) {
      // If the word is longer than 10 characters, print its abbreviation.
      // s[0] is the first character.
      // s.size() - 2 is the number of characters between the first and last.
      // s.back() is the last character.
      cout << s[0] << s.size() - 2 << s.back() << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << s << endl;
    }
  }
  return 0;
}
