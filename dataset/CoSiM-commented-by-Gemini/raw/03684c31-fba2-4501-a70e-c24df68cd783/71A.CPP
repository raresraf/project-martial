/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem that shortens long strings.
 *
 * This program reads a series of strings and, for any string longer than 10 characters,
 * it prints an abbreviation consisting of the first character, the number of characters
 * between the first and last, and the last character. Shorter strings are printed as-is.
 *
 * Algorithm: String abbreviation based on length.
 * Time Complexity: O(T * L) where T is the number of test cases and L is the max length of a string (dominated by I/O).
 * Space Complexity: O(L) for storing each string.
 */
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  // Variable to store the number of test cases.
  int t;
  // Read the number of test cases from standard input.
  cin >> t;
  
  /**
   * @brief Main loop to process each test case.
   * Pre-condition: `t` contains the number of remaining test cases.
   * Invariant: The loop will execute `t` times.
   */
  while (t--) {
    string s;
    // Read the input string for the current test case.
    cin >> s;
    int size = s.length();

    /**
     * @brief Conditional block to check string length.
     * Logic: If the string's length is greater than 10, it is abbreviated.
     * Otherwise, it is printed in its original form.
     */
    if (size > 10) {
      // Print the abbreviated string format: first char, length-2, last char.
      cout << s[0] << (size - 2) << s[size - 1] << endl;
    } else {
      // Print the string as is because it's not "too long".
      cout << s << endl;
      ;
    }
  }
}
