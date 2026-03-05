/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (Codeforces 281A).
 *
 * This program reads a single word and capitalizes its first letter,
 * leaving the rest of the word unchanged. The modified word is then
 * printed to standard output.
 *
 * Algorithm: String traversal and character manipulation. It iterates through
 * the characters of the string, capitalizing only the first one.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a single word from standard input, capitalizes its first character,
 * and then prints the modified word to standard output.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties
  // cin from cout to optimize high-throughput I/O required for competitive programming constraints.
  ios::sync_with_stdio(0);
  cin.tie(0);

  // Declare a string 's' to store the input word.
  string s;
  // Read the input word into 's'.
  cin >> s;

  /**
   * @brief Block Logic: Iterates through the characters of the word to capitalize the first.
   *
   * Pre-condition: 's' contains the input word.
   * Invariant: Only the character at index 0 is potentially changed; other characters remain as is.
   */
  for (int i = 0; i < s.length(); i++) {
    // Pre-condition: Check if the current character is the first character of the word.
    if (i == 0) {
      // If it's the first character, convert it to uppercase and print.
      cout << char(toupper(s[i]));
    } else {
      // For all other characters, print them as they are.
      cout << s[i];
    }
  }

  // Print a newline character after the word.
  cout << endl;

  return 0;
}
