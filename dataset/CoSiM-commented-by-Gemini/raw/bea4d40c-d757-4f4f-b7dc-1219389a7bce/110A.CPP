/**
 * @file 110A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 110A "Nearly Lucky Number".
 *
 * @details
 * This program determines if a given number is "nearly lucky". A number is considered
 * nearly lucky if the total count of its "lucky digits" ('4' and '7') is itself
 * a lucky number. The only lucky numbers that the count can be are 4 or 7, given
 * the constraints on the input number's length.
 *
 * @algorithm
 * 1. Read the input number as a string to handle potentially large values.
 * 2. Iterate through each character of the string.
 * 3. Count the occurrences of the characters '4' and '7'.
 * 4. After counting, check if the final count is equal to 4 or 7.
 * 5. If it is, the number is nearly lucky ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * - Time Complexity: O(L), where L is the number of digits in the input number,
 *   as the program performs a single pass on the string.
 * - Space Complexity: O(L) to store the input number as a string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  string n;
  int count = 0;
  // Read the number as a string to handle large inputs (up to 18 digits).
  cin >> n;

  /**
   * @brief Counts the number of lucky digits ('4' or '7') in the string.
   */
  for (int i = 0; i < n.length(); i++) {
    /**
     * @note This line converts the character digit to its integer equivalent
     * by subtracting the ASCII value of '0' (48). While it works, a more
     * idiomatic way is `n[i] - '0'`, or to simply compare the character
     * `n[i]` with '4' and '7'.
     */
    n[i] = n[i] - 48;

    if ((n[i] == 4 || n[i] == 7)) {
      count++;
    }
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * Given the input constraints (max 18 digits), the count can never be
   * 47 or 74. The only lucky numbers the count could be are 4 or 7.
   */
  if (count == 4 || count == 7) // The checks for 47 and 74 are superfluous.
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
