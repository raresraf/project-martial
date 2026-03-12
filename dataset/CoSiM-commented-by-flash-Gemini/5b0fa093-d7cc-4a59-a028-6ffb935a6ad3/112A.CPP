/**
 * @5b0fa093-d7cc-4a59-a028-6ffb935a6ad3/112A.CPP
 * @brief This program solves problem 112A ("Petya and Strings").
 * It performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then uses `std::string::compare`
 * to determine their relative order, outputting -1, 1, or 0.
 *
 * Algorithm: Two-pass approach: first pass converts strings to lowercase using `std::transform`,
 * second pass compares them using `std::string::compare`.
 * Time Complexity: O(L) where L is the length of the strings.
 * Space Complexity: O(1) beyond the input strings (modifications are in-place).
 */

#include <algorithm> // For std::transform.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <string>    // For using std::string.
#include <cctype>    // For ::tolower.
using namespace std; // Uses the standard namespace.

int main() {
  string s1, s2; // Declares two string variables to store the input words.
  cin >> s1;     // Reads the first string from standard input.
  cin >> s2;     // Reads the second string from standard input.

  // Block Logic: Converts the entire first string to lowercase in-place.
  // `std::transform` applies the `::tolower` function to each character in the range.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  // Block Logic: Converts the entire second string to lowercase in-place.
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  /**
   * Block Logic: Compares the two lowercase strings lexicographically using `std::string::compare()`.
   * The `compare()` method returns:
   *   - a positive value if the first string is lexicographically greater.
   *   - a negative value if the first string is lexicographically smaller.
   *   - zero if the strings are equal.
   * Invariant: Prints 1, -1, or 0 based on the comparison result.
   */
  if (s1.compare(s2) > 0) // Inline: Checks if s1 is lexicographically greater than s2.
    cout << "1" << endl;   // Prints "1".
  else if (s1.compare(s2) < 0) // Inline: Checks if s1 is lexicographically less than s2.
    cout << "-1" << endl;    // Prints "-1".
  else // Inline: If s1.compare(s2) returns 0, meaning strings are equal.
    cout << "0" << endl;     // Prints "0".

  return 0; // Indicates successful program termination.
}
