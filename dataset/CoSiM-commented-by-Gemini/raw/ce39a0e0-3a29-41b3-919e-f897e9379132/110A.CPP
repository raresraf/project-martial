/**
 * @file 110A.CPP
 * @brief Solution for competitive programming problem "Nearly Lucky Number".
 * @details This program determines if a given number 'n' is "nearly lucky".
 *          A number is nearly lucky if the count of its lucky digits ('4' and '7')
 *          is itself a lucky number. A lucky number is a positive integer
 *          composed only of the digits '4' and '7'.
 *
 * Algorithm:
 * 1. Count the number of '4' and '7' digits in the input number `n`.
 * 2. Check if this count is a positive number composed solely of '4' and '7' digits.
 *
 * Time Complexity: O(log(n)), as the number of digits in `n` is proportional to log10(n).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  long long int n, cnt = 0;
  bool lucky = true;
  cin >> n;

  /**
   * @brief First pass: Count the occurrences of lucky digits ('4' and '7')
   *        in the input number `n`.
   * @invariant `cnt` holds the number of lucky digits found in the processed
   *            part of `n`. `n` is progressively reduced by a factor of 10.
   */
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7)
      cnt++;
    n /= 10;
  }

  /**
   * @brief Second pass: Check if the count `cnt` is itself a lucky number.
   *        A lucky number must be positive and contain only '4' and '7'.
   * @note There is a bug here: if `cnt` is 0, this loop is skipped and the
   *       program incorrectly outputs "YES" because `lucky` remains true.
   *       A count of 0 is not a lucky number.
   * @invariant `lucky` is true if all digits of `cnt` processed so far are
   *            '4' or '7'. `cnt` is progressively reduced.
   */
  while (cnt > 0) {
    if (cnt % 10 != 4 && cnt % 10 != 7) {
      lucky = false;
      break;
    }
    cnt /= 10;
  }

  // A number is nearly lucky if the flag remains true.
  if (lucky)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
