#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
typedef long long ll;

/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 * @details This program checks if a given number is "nearly lucky". A number is defined
 * as nearly lucky if the count of its "lucky digits" ('4' and '7') is itself a
 * lucky number (i.e., the count is 4 or 7).
 *
 * Algorithm:
 * The input number is read as a string to handle its large potential size. The program
 * then iterates through each character of the string, maintaining a counter for the
 * number of times '4' or '7' appears. After the loop, it checks if this final counter
 * value is equal to 4 or 7. If it is, the original number is nearly lucky, and "YES"
 * is printed. Otherwise, "NO" is printed.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number, because
 * it involves a single pass over the string.
 * Space Complexity: O(L) to store the number as a string.
 */
int main() {
  string s;
  cin >> s;
  ll len = s.length();
  ll count = 0;

  /**
   * @brief Count the number of lucky digits ('4' or '7') in the string.
   */
  for (int i = 0; i < len; i++) {
    if (s[i] == '7' || s[i] == '4') {
      count++;
    }
  }

  /**
   * @brief Determine if the count of lucky digits is itself a lucky number.
   * For a number with up to 18 digits, the only possible lucky counts are 4 and 7.
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
