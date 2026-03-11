/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 *
 * This program reads a single word, capitalizes its first letter, and then prints the modified word.
 * Algorithm: String manipulation with manual ASCII-based character case conversion.
 * Time Complexity: O(L) where L is the length of the input string, due to string input and output,
 *                  but the core logic for capitalization is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a word, converts its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  string a;
  // Block Logic: Read the input word from standard input.
  // Pre-condition: User provides a single word (string).
  // Invariant: 'a' will contain the input word.
  cin >> a;

  // Block Logic: Capitalize the first character of the string if it is a lowercase letter.
  // The condition checks if the first character is NOT already an uppercase letter.
  // If it's a lowercase letter, adding ('A' - 'a') (which is -32) converts it to uppercase.
  // Pre-condition: 'a' is not empty.
  // Invariant: The first character of 'a' will be uppercase if it was a lowercase letter.
  if (!('A' <= a[0] && a[0] <= 'Z'))
    a[0] += 'A' - 'a';

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The word with its first letter capitalized is displayed, followed by a newline.
  cout << a << endl;
  return 0;
}
