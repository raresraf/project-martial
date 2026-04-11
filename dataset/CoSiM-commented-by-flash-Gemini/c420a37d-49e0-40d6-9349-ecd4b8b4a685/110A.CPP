/**
 * @file 110A.CPP
 * @brief This program determines if a given number (represented as a string) is "almost lucky".
 * A number is considered almost lucky if the count of '4's and '7's within its digits
 * is exactly 4 or exactly 7.
 *
 * Algorithm: String iteration and character counting.
 * Time Complexity: O(L) where L is the number of digits in the input number string.
 * Space Complexity: O(L) to store the number string.
 */

#include <algorithm> // Required for std::count
#include <iostream>
#include <string>    // Required for std::string
using namespace std;

int main() {
  // Functional Utility: 'x' stores the input number as a string.
  string x;
  // Block Logic: Reads the number as a string from standard input.
  cin >> x;

  // Functional Utility: 'n' stores the length of the input number string (unused after calculation).
  int n = x.length();
  // Functional Utility: 'i' stores the total count of '4's and '7's in the number string.
  int i;

  // Functional Utility: Counts occurrences of '4' and '7' in the string 'x'.
  // Inline: `count(x.begin(), x.end(), '4')` counts '4's, and similar for '7'.
  i = count(x.begin(), x.end(), '4') + count(x.begin(), x.end(), '7');

  /**
   * Block Logic: Checks if the total count of '4's and '7's is either 4 or 7.
   * Pre-condition: 'i' contains the aggregated count of '4's and '7's.
   * Invariant: If the count is 4 or 7, prints "YES"; otherwise, prints "NO".
   * Inline: `i == 4 | i == 7` uses bitwise OR. While `||` (logical OR) is typically intended for booleans,
   *         `|` (bitwise OR) also works here because the comparisons `i == 4` and `i == 7`
   *         evaluate to 0 or 1, and the result is implicitly converted to boolean.
   */
  if (i == 4 | i == 7) // Note: Bitwise OR used instead of logical OR.
    cout << "YES";
  else
    cout << "NO";
}
