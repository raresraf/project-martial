#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

/**
 * @file 110A.CPP
 * @brief Solution to a competitive programming problem, "Nearly Lucky Number".
 *
 * Problem Statement:
 * A "lucky number" is a positive integer whose decimal representation contains
 * only the digits 4 and 7. For example, 47, 744, 4 are lucky numbers, while
 * 5, 17, 467 are not.
 * A number is called "nearly lucky" if the count of lucky digits ('4' or '7')
 * in it is a lucky number.
 *
 * This program determines if a given number is a nearly lucky number.
 *
 * Algorithm:
 * 1. Read the input number `n`.
 * 2. Initialize a counter `c` to zero.
 * 3. Iterate through each digit of `n`.
 * 4. For each digit, if it is a '4' or a '7', increment the counter `c`.
 * 5. After iterating through all digits, check if the final count `c` is
 *    itself a lucky number. Since the input number has at most 18 digits, the
 *    count `c` will be small. The only single or double-digit lucky numbers
 *    are 4, 7, 44, 47, etc. Given the constraint, we only need to check if
 *    the count is 4 or 7.
 * 6. Print "YES" if the count is 4 or 7, otherwise print "NO".
 *
 * Time Complexity: O(log10(N)), where N is the input number, as we process
 * each digit once.
 * Space Complexity: O(1).
 */

using namespace std;

typedef long long ll;

int main() {
  ll n;
  int c = 0; // Counter for the number of lucky digits (4s and 7s).

  // A compact for-loop to process the number:
  // 1. Initialization: `cin >> n` reads the number.
  // 2. Condition: `n > 0` continues the loop until all digits are processed.
  // 3. Iteration: `n /= 10` removes the last digit.
  for (cin >> n; n > 0; n /= 10) {
    ll d = n % 10; // Get the last digit.

    // This is a concise way to increment the counter.
    // The boolean expression (d == 4 || d == 7) evaluates to 1 (true) or
    // 0 (false), which is then added to 'c'.
    c += (d == 4 || d == 7);
  }

  // Pre-condition: Check if the count of lucky digits is itself a lucky number.
  // For the given constraints, the only possible lucky counts are 4 and 7.
  if (c == 4 || c == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}