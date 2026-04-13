/**
 * @file 236A.CPP
 * @brief This program determines a message ("CHAT WITH HER!" or "IGNORE HIM!")
 * based on the parity (odd or even) of the number of unique characters in an input username string.
 * A user chat message is determined by counting unique characters: even count for "CHAT WITH HER!", odd for "IGNORE HIM!".
 * Algorithm: Attempts to count unique characters by comparing each character with subsequent ones and
 * decrementing a total count if a duplicate is found. Note: The unique character counting logic in this
 * implementation is potentially flawed and may not always yield the correct number of unique characters
 * for all input strings.
 * Time Complexity: O(N^2) due to nested loops for duplicate checking, where N is the string length.
 * Space Complexity: O(N) for storing the input string.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for std::string.

using namespace std;

int main() {
  string str; // Stores the input username string.
  // Reads the input username string from standard input.
  cin >> str;

  string strTest = str; // Creates a copy of the input string for comparison.
  // Initializes 'cnt' with the total length of the string, intending to decrement for duplicates.
  // This assumes all characters are unique initially.
  int cnt = str.length();
  unsigned int i = 0; // Loop index for the outer loop.

  /**
   * Block Logic: Iterates through each character of the input string ('str')
   * and attempts to identify and subtract duplicate characters from 'cnt'.
   * For each character `str[i]`, it checks all subsequent characters `strTest[j]`
   * (which is initially the same as `str`). If a match is found, `cnt` is decremented
   * and the inner loop breaks. This logic is problematic because it only decrements
   * `cnt` once for each `str[i]` if a duplicate is found, regardless of how many times
   * that character appears later, and it also doesn't reset for new unique characters.
   * Invariant: `i` progresses through the input string.
   */
  while (i != str.length()) {
    for (unsigned int j = i + 1; j < str.length(); j++) { // Corrected inner loop condition to str.length()
      if (str[i] == strTest[j]) { // Compares current character with subsequent characters in the copy.
        cnt--;                    // Decrements count if a duplicate is found.
        break;                    // Breaks from inner loop, moving to the next character in 'str'.
      }
    }

    i++; // Moves to the next character in the outer loop.

    /**
     * Conditional Logic: After the main loop completes (when 'i' reaches `str.length()`),
     * this block determines and prints the output message based on the parity of 'cnt'.
     * If 'cnt' is even, it prints "CHAT WITH HER!"; otherwise, "IGNORE HIM!".
     * This logic is executed only once after the entire processing.
     */
    if (i == str.length()) { // This condition executes only after the outer while loop finishes.
      if (cnt % 2 == 0)
        cout << "CHAT WITH HER!" << endl;
      else
        cout << "IGNORE HIM!" << endl;
    }
  }
  // Standard program exit.
  return 0;
}
