/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem (Codeforces 110A).
 * @details A "nearly lucky number" is one where the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number. This program determines if a given number is nearly lucky.
 *
 * Algorithm: Digit Counting
 * 1. The input number is read as a string to easily iterate over its digits.
 * 2. The `std::count` algorithm is used to find the total number of '4's and '7's.
 * 3. This total count is then checked to see if it is a lucky number (i.e., if the count is 4 or 7).
 *    Given the constraints on the input number's length, these are the only possible lucky counts.
 *
 * NOTE: The code uses a bitwise OR `|` in the final condition `if (i == 4 | i == 7)`.
 * While it works correctly in this specific case because the results of the comparisons
 * are 0 or 1, the standard and correct operator for this logic is the logical OR `||`.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * as `std::count` needs to iterate through the string.
 * Space Complexity: O(L) to store the input number as a string.
 */

#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  // x: stores the input number as a string, allowing for digit-by-digit access.
  string x;
  cin >> x;

  // n: unused variable storing the length of the string.
  int n = x.length();
  // i: stores the total count of lucky digits ('4' and '7').
  int i;

  // Core Logic: Count the occurrences of '4' and '7' in the input string.
  i = count(x.begin(), x.end(), '4') + count(x.begin(), x.end(), '7');

  /**
   * Final Check: Determine if the count of lucky digits is itself a lucky number.
   * The condition checks if the count 'i' is 4 or 7.
   * Note the use of bitwise OR `|` instead of the more appropriate logical OR `||`.
   */
  if (i == 4 | i == 7)
    cout << "YES";

  else
    cout << "NO";
}