/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic.
 * Words longer than 10 characters are abbreviated to their first letter,
 * followed by the count of characters between the first and last letters,
 * and then the last letter. Words 10 characters or shorter remain unchanged.
 * This is a common string manipulation problem in competitive programming.
 *
 * Algorithm: String length check and formatting.
 * 1. Read the number of words (test cases) `t`.
 * 2. For each word:
 *    a. Read the word `s`.
 *    b. Check its length.
 *    c. If the length is greater than 10, print the abbreviated form.
 *    d. Otherwise, print the word as is.
 * Time Complexity: O(Sum of all word lengths) where each character is processed at most once.
 * Space Complexity: O(Max_Word_Length) to store the current word.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads test cases, applies word abbreviation logic, and prints the result.
 * @pre Input consists of an integer `t` (number of test cases), followed by `t` word strings.
 * @post For each word, prints its abbreviated form (if applicable) to standard output.
 * @return 0 upon successful execution.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Original commented line, preserved.

  // Functional Utility: 't' for the number of test cases.
  int t;
  // Functional Utility: 's' to store the current word.
  string s;
  // Block Logic: Loops through each test case (`t` times).
  // Invariant: `t` decrements, indicating remaining test cases.
  for (scanf("%d", &t); t--;) {
    cin >> s; // Functional Utility: Reads the current word.
    // Block Logic: Checks if the word's length exceeds 10 characters to determine if abbreviation is needed.
    // Pre-condition: `s` contains the current word.
    if (s.size() <= 10)
      cout << s << endl; // Functional Utility: Prints the word as is if it's 10 characters or shorter.
    else
      // Functional Utility: Prints the abbreviated form: first char + (length - 2) + last char.
      cout << s[0] << (s.size() - 2) << s[s.size() - 1] << endl;
  }
  return 0;
}
