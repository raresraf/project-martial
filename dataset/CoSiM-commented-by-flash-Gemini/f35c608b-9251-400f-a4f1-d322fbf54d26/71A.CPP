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
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.

using namespace std;

/**
 * @brief Main function to process a series of words and apply abbreviation rules.
 * Functional Utility: Handles multiple test cases, reading a string for each and applying the abbreviation logic.
 */
int main() {
  string a; // Variable to store the input string.
  int n;    // Variable to store the number of test cases.
  // Block Logic: Reads the number of test cases 'n'.
  cin >> n;
  /**
   * Block Logic: Iterates through each test case.
   * Pre-condition: 'n' is a non-negative integer representing the number of words to process.
   * Invariant: In each iteration, one word is read and processed according to its length.
   * Inline: `n--` decrements the counter and evaluates to true as long as `n` is greater than 0.
   */
  while (n--) {
    // Block Logic: Reads the current string for processing.
    cin >> a;
    /**
     * Block Logic: Applies the abbreviation rule based on string length.
     * If the string length is greater than 10, it constructs an abbreviated string.
     * Otherwise, it prints the original string.
     * Inline: `a.length()` gets the length of the string.
     * Inline: `a[0]` accesses the first character.
     * Inline: `a.back()` accesses the last character.
     * Inline: `"\n"` is used for a newline, often preferred over `std::endl` in competitive programming
     *         for minor performance gains as `std::endl` also flushes the buffer.
     */
    if (a.length() > 10)
      cout << a[0] << (a.length() - 2) << a.back() << "\n";
    else
      cout << a << "\n";
  }
}
