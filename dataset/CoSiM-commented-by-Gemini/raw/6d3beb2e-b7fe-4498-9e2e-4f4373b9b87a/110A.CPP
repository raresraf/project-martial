
#include <iostream>

using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution to Codeforces problem 110A, "Nearly Lucky Number".
 * @details The program determines if a given number is "nearly lucky". A number
 * is nearly lucky if the count of its lucky digits ('4' and '7') is itself a
 * lucky number (i.e., the count is 4 or 7).
 *
 * Algorithm:
 * The input number is read as a string to handle potentially large values. The
 * program then iterates through each character of the string, counting the
 * occurrences of the lucky digits '4' and '7'. After counting, it checks if
 * the total count is equal to 4 or 7. If it is, the number is classified as
 * nearly lucky.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * as it requires a single pass over the string.
 * Space Complexity: O(L) to store the number as a string.
 */
int main() {
  // s: the input number read as a string.
  string s;
  cin >> s;
  int sz = s.length();
  
  // cnt: counter for the number of lucky digits found.
  int cnt = 0;

  /**
   * @brief Count the occurrences of '4' and '7' in the string.
   */
  for (int i = 0; i < sz; ++i) {
    if (s[i] == '4' || s[i] == '7')
      cnt++;
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * For a number with up to 18 digits, the only possible lucky counts are 4 and 7.
   */
  if (cnt == 4 || cnt == 7)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}
