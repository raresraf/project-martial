/**
 * @file 110A.CPP
 * @brief Determines if a number is a "nearly lucky number".
 *
 * A "nearly lucky number" is defined as a number for which the count of
 * "lucky digits" (4 and 7) is itself a lucky number (i.e., the count is 4 or 7).
 * This program reads an integer and checks if it meets this criterion.
 *
 * Algorithm:
 * 1. Read a long long integer `n`.
 * 2. Initialize a counter `c` to 0 to store the count of lucky digits.
 * 3. Make a copy of the input number `n` to iterate through its digits without
 *    modifying the original.
 * 4. Loop while the copied number is greater than 0:
 *    a. Extract the last digit using the modulo operator (`% 10`).
 *    b. If the digit is either 4 or 7, increment the counter `c`.
 *    c. Remove the last digit by integer division (`/= 10`).
 * 5. After the loop, check if the final count `c` is equal to 4 or 7.
 * 6. If it is, the number is "nearly lucky", so print "YES".
 * 7. Otherwise, print "NO".
 *
 * Time Complexity: O(log10(N)), where N is the input number, because the
 * loop runs once for each digit of the number.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
int main() {
  ll n;
  cin >> n;
  ll k = n, r, c = 0;
  
  /**
   * @brief Loop to count the occurrences of lucky digits (4 and 7).
   * Pre-condition: `k` holds the number to be processed.
   * Invariant: `c` holds the count of lucky digits for the digits of `k`
   * that have been processed so far.
   */
  while (k > 0) {
    r = k % 10; // Get the last digit.
    if (r == 4 || r == 7)
      c++;
    k /= 10; // Remove the last digit.
  }
  
  // Check if the count of lucky digits is itself a lucky number (4 or 7).
  if (c == 4 || c == 7)
    cout << "YES\n";
  else
    cout << "NO\n";
    
  return 0;
}