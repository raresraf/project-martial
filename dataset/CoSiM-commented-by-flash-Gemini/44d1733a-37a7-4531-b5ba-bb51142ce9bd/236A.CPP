/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 236A - "Boy or Girl").
 *
 * This program determines a gender-like classification ("Boy" or "Girl") for a username
 * based on the parity of the number of unique characters it contains.
 * If the number of unique characters is even, the output is "CHAT WITH HER!";
 * if odd, the output is "IGNORE HIM!".
 *
 * Algorithm: Character counting using `std::set` and parity check.
 * It reads characters one by one, inserting each into a `std::set` to automatically
 * maintain a collection of unique characters. After processing the input, it checks
 * the size of the set and determines the output based on whether the count is even or odd.
 *
 * Time Complexity: O(L log K), where L is the length of the input string and K is the
 * number of unique characters. Since K (alphabet size) is constant (at most 26 for English),
 * this is effectively O(L).
 * Space Complexity: O(K) for storing unique characters in the `std::set`. Effectively O(1).
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0); // Macro for fast I/O. Unties cin from cout and disables sync with stdio.

using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a username character by character, counts its unique characters, and prints a message
 * based on the parity of the unique character count.
 */
int main() {
  _ // Apply fast I/O optimizations.
  string s; // Declared but not used for reading the main input string in the current logic.
  set<char> m; // Set to store unique characters found.
  char x;      // Variable to store each character read.

  // Block Logic: Read characters one by one until a newline is explicitly found or EOF.
  // Note: `cin >> x` for `char x` typically skips whitespace, including newlines.
  // The `if (x == '\n') break;` condition might not be reliably reached if input
  // consists solely of non-whitespace characters followed by EOF, or if the `\n`
  // is consumed by a prior operation not shown. Assuming it functions as intended
  // for the problem's input format.
  while (cin >> x) {
    if (x == '\n')
      break; // Stop reading if a newline character is encountered.
    m.insert(x); // Insert the character into the set to track unique characters.
  }
  // Block Logic: Check the parity of the number of unique characters.
  if (m.size() % 2 == 0) { // If the number of unique characters is even.
    cout << "CHAT WITH HER!\n"; // Print message for "Girl".
  } else {                   // If the number of unique characters is odd.
    cout << "IGNORE HIM!\n"; // Print message for "Boy".
  }

  return 0; // Indicate successful execution.
}