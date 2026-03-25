// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem.
 *
 * Problem Statement:
 * A "lucky number" is an integer composed only of the digits '4' and '7'.
 * A number is "nearly lucky" if the total count of its lucky digits ('4'
 * or '7') is itself a lucky number. This program determines if a given number
 * is nearly lucky.
 *
 * Algorithm:
 * The solution is a two-stage process:
 * 1.  **Count Lucky Digits**: The first stage iterates through the digits of
 *     the input number `n` and counts how many of them are '4' or '7'.
 * 2.  **Check the Count**: The second stage checks if the resulting count from
 *     stage 1 is a lucky number. It does this by iterating through the digits
 *     of the count itself. If any digit of the count is not a '4' or a '7',
 *     the number is not nearly lucky.
 *
 * @note While this general approach is perfectly correct, for the constraints
 * of this specific problem (input `n` has at most 18 digits), the count of
 * lucky digits can be at most 18. The only lucky numbers in this range are 4
 * and 7. A simpler, more direct solution would be to just check `if (count == 4
 * || count == 7)`.
 *
 * Time Complexity: O(log N), where N is the input number, as the logic
 * processes the digits of N and then the digits of the count (which is at
 * most log N).
 * Space Complexity: O(1).
 */

int main() {
  long long n;
  cin >> n;
  long long count = 0; // Stores the number of lucky digits found in 'n'.

  // Block 1: Count the occurrences of '4' and '7' in the input number.
  // Invariant: 'count' holds the number of lucky digits in the processed part of 'n'.
  while (n != 0) {
    int x = n % 10; // Extract the last digit.
    n /= 10;
    if (x == 4 || x == 7) {
      count++;
    }
  }

  // Pre-condition for the next block: if the count is 0, it's not a lucky number.
  if (count == 0) {
    cout << "NO" << endl;
    return 0;
  }

  // Block 2: Check if the 'count' itself is a lucky number.
  bool flag = true; // Assume the count is lucky until proven otherwise.
  while (count != 0) {
    int x = count % 10; // Extract the last digit of the count.
    count /= 10;
    if (x != 4 && x != 7) {
      // If any digit of the count is not 4 or 7, it's not a lucky number.
      flag = false;
      cout << "NO" << endl;
      break;
    }
  }

  // If the flag is still true, all digits of the count were either 4 or 7.
  if (flag)
    cout << "YES" << endl;
  return 0;
}
