/**
 * @file 112A.CPP
 * @brief Solution for a lexicographical string comparison problem using STL algorithms.
 * @details This program reads two strings and compares them lexicographically in a
 * case-insensitive manner. It uses the `std::transform` function to convert strings
 * to lowercase and `std::string::compare` for the comparison logic.
 *
 * @algorithm Case-insensitive lexicographical comparison.
 * @complexity
 * - Time: O(L), where L is the length of the strings. `std::transform` is O(L) and
 *   `std::string::compare` is also O(L).
 * - Space: O(L), to store the two input strings.
 */

#include <algorithm> // For std::transform
#include <iostream>  // For std::cin, std::cout
#include <string>    // For std::string

// Using the standard namespace to avoid prefixing std:: to standard library elements.
using namespace std;

int main() {
  // Declare two strings to hold the input.
  string s1, s2;
  // Read the two strings from standard input.
  cin >> s1;
  cin >> s2;

  /**
   * @brief Normalize both strings to lowercase for case-insensitive comparison.
   * `std::transform` applies the `::tolower` function to each character of the
   * string, from its beginning to its end, and writes the result back in-place.
   */
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  /**
   * @brief Compare the normalized strings and output the result.
   * The `std::string::compare` method returns:
   *  - a value > 0 if the first string is lexicographically greater.
   *  - a value < 0 if the first string is lexicographically smaller.
   *  - 0 if the strings are equal.
   */
  // Pre-condition: Check if s1 is greater than s2.
  if (s1.compare(s2) > 0)
    cout << "1" << endl;
  // Pre-condition: Check if s1 is less than s2.
  else if (s1.compare(s2) < 0)
    cout << "-1" << endl;
  // Invariant: If neither greater nor less, the strings must be equal.
  else
    cout << "0" << endl;

  return 0;
}
