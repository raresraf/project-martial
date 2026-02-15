/**
 * @file 110A.CPP
 * @brief Determines if a number is "nearly lucky".
 *
 * A "nearly lucky" number is defined as a number where the count of its "lucky"
 * digits ('4' and '7') is itself a "lucky number". A "lucky number" is a
 * positive integer composed only of the digits '4' and '7'.
 *
 * @algorithm
 * 1. Read the input number as a string.
 * 2. Count the occurrences of the digits '4' and '7' in the string.
 * 3. Check if the resulting count is a "lucky number" (i.e., composed only of digits '4' and '7').
 * 4. Print "YES" if the count is a non-zero lucky number, otherwise print "NO".
 *
 * @complexity
 * Time: O(L), where L is the number of digits in the input string. The check on the count is logarithmic in the count's value, which is significantly smaller than L.
 * Space: O(L) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // `a` is declared but not used.
  int a;
  /**
   * @var flag A boolean flag to indicate if a non-lucky digit is found in the count.
   *           0 means the count is lucky so far, 1 means it is not.
   */
  int flag = 0;
  // @var s The input number read as a string to easily iterate over its digits.
  string s;
  cin >> s;
  /**
   * @var cnt Stores the count of lucky digits ('4' or '7') in the input string.
   */
  int cnt = 0;

  /**
   * @brief Counts the number of lucky digits in the input string.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      cnt++;
    }
  }

  int rem;
  /**
   * @brief Checks if the count of lucky digits (`cnt`) is itself a lucky number.
   */
  if (cnt > 0) {
    /**
     * @brief Iterates through each digit of `cnt`.
     *
     * This loop extracts the last digit of `cnt` using the modulo operator (`rem = cnt % 10`),
     * then checks if it's a '4' or a '7'. If any digit is not '4' or '7', the `flag` is
     * set to 1, and the number is not lucky.
     */
    while (cnt > 0) {
      rem = cnt % 10;
      cnt = cnt / 10;
      if (rem != 4 && rem != 7) {
        flag = 1;
        break;
      }
    }

    // @brief Prints the result based on the flag.
    if (flag == 0) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  } else {
    // If the count of lucky digits is zero, it's not a nearly lucky number.
    cout << "NO";
  }
}

