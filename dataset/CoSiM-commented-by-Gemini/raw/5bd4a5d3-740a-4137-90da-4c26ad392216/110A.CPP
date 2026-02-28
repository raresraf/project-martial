/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem from Codeforces.
 * @details A number is considered "nearly lucky" if the number of its lucky digits
 * (4 and 7) is itself a lucky number (i.e., the count is 4 or 7). This program
 * reads an integer, counts its lucky digits, and then determines if that count
 * makes the number "nearly lucky".
 *
 * @algorithm Digit Counting
 * 1. Read the input number `n`.
 * 2. Initialize a counter for lucky digits to zero.
 * 3. Iterate through each digit of `n` by repeatedly taking the number modulo 10
 *    (to get the last digit) and then dividing by 10 (to remove the last digit).
 * 4. If a digit is 4 or 7, increment the counter.
 * 5. After the loop, check if the final value of the counter is 4 or 7.
 * 6. If it is, print "YES"; otherwise, print "NO".
 *
 * @complexity
 * - Time: O(log10(N)), where N is the input number. The number of iterations in the
 *   while loop is equal to the number of digits in N.
 * - Space: O(1), for storing a few integer variables.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

int main() {
  // n: the input number.
  // m: used to store each digit of n temporarily.
  // lucky_digit_count: counts the occurrences of lucky digits (4 and 7).
  long long int n;
  int m, lucky_digit_count = 0;

  // Read the input number.
  cin >> n;

  /**
   * @brief Count the number of lucky digits (4 or 7) in n.
   * The loop continues as long as there are digits left to process in n.
   */
  // Invariant: `lucky_digit_count` holds the count of lucky digits from the
  // portion of the number already processed. `n` holds the remaining digits.
  while (n != 0) {
    // Extract the last digit of n.
    m = n % 10;
    // Remove the last digit from n.
    n = n / 10;

    // Pre-condition: Check if the extracted digit is a lucky digit.
    if (m == 4 || m == 7) {
      lucky_digit_count++;
    }
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * A "nearly lucky number" has a count of lucky digits that is either 4 or 7.
   */
  if (lucky_digit_count == 4 || lucky_digit_count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0; // Indicate successful execution.
}
