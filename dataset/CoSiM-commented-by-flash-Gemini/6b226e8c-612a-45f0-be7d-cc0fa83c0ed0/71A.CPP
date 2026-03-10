/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic, commonly found in competitive programming.
 *   If a word's length exceeds 10 characters, it is abbreviated to its first letter, followed by
 *   the number of characters between the first and last letters, and then its last letter.
 *   Otherwise, the word is printed as is.
 *   Problem description: "Way Too Long Words", Codeforces 71A.
 * * Algorithm: String length check and conditional abbreviation.
 * Time Complexity: O(N * L_max), where N is the number of test cases and L_max is the maximum string length.
 * Space Complexity: O(L_max) to store the longest input string.
 */
#include <cstdio>   // Included for C-style I/O functions, though not directly used in the main logic.
#include <iostream> // Required for std::cin, std::cout, std::endl, std::getline.
#include <string>   // Required for std::string, std::to_string.

using namespace std;

/**
 * @brief Main function to process a series of words and apply abbreviation rules.
 * Functional Utility: Handles multiple test cases, reading a string for each and applying the abbreviation logic.
 */
int main() {
  int i, n; // `i` for loop counter, `n` for number of test cases.
  // Block Logic: Reads the number of test cases 'n'.
  // Note: When mixing `cin >> n` with `getline(cin, s)`, `cin >> n` leaves a newline character
  // in the input buffer. The first call to `getline(cin, s)` after `cin >> n` will consume
  // this leftover newline, resulting in an empty string `s`. A common fix is to add `cin.ignore()`
  // after `cin >> n` or use `getline` for all input.
  cin >> n;
  /**
   * Block Logic: Iterates through each test case from 0 to 'n'.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: In each iteration, one word is read and processed according to its length.
   * Note: The loop condition `i <= n` will cause `n+1` iterations. This might be off-by-one
   *       depending on the problem's exact test case count interpretation.
   */
  for (i = 0; i <= n; i++) {
    string s; // Variable to store the current input string.

    // Block Logic: Reads the entire line (including spaces) into string `s`.
    // Due to mixing `cin >> n` and `getline`, the first `s` will be empty.
    getline(cin, s);
    /**
     * Conditional Logic: Applies the abbreviation rule based on string length.
     * If the string length is 10 or less, it prints the original string.
     */
    if (s.size() <= 10) {
      cout << s << endl;
    }
    /**
     * Conditional Logic: If the string length is greater than 10, it constructs an abbreviated string.
     * Inline: `s.size()` gets the length of the string.
     * Inline: `s[0]` accesses the first character.
     * Inline: `s[s.size() - 1]` accesses the last character.
     * Inline: `to_string(s.size() - 2)` converts the calculated integer length to a string.
     * The `s3` string is built by concatenating these parts.
     */
    else if (s.size() > 10) {
      string s2 = ""; // This variable `s2` is declared but not used.
      string s3 = s[0] + to_string(s.size() - 2) + s[s.size() - 1];
      cout << s3 << endl;
    }
  }
  return (0);
}