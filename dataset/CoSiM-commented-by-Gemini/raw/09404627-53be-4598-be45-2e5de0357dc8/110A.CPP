/**
 * @file 110A.CPP
 * @brief Determines if a number is a "Nearly Lucky Number".
 * @details This program solves the "Nearly Lucky Number" problem from Codeforces.
 * A number is "nearly lucky" if the count of its "lucky digits" (4 and 7) is
 * itself a lucky number (i.e., 4 or 7).
 *
 * @see https://codeforces.com/problemset/problem/110/A
 *
 * Algorithm:
 * 1. Count the occurrences of the digits '4' and '7' in the input number.
 *    This is done by repeatedly taking the number modulo 10 to get the last
 *    digit, and then dividing by 10 to remove the last digit, until the number
 *    becomes zero.
 * 2. After counting, check if the count itself is a lucky number. Given the
 *    constraints (input number up to 10^18 has at most 18 digits), the only
 *    possible lucky numbers the count can be are 4 or 7.
 * 3. If the count is 4 or 7, print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(log10(N)), where N is the input number, as the program
 * iterates once for each digit in N.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  long long n;
  int count; // Stores the number of lucky digits (4s and 7s).
  
  count = 0;
  cin >> n;

  /**
   * @brief Main loop to count the lucky digits in the input number.
   *
   * Pre-condition: 'n' holds the input number.
   * Invariant: After each iteration, one digit of 'n' has been processed,
   *            and 'count' is updated if that digit was lucky.
   */
  while (n != 0) {
    // Get the last digit of the number.
    int rem = n % 10;
    // Remove the last digit from the number.
    n = n / 10;
    
    // Check if the digit is a lucky digit.
    if (rem == 4 || rem == 7) {
      count++;
    }
  }
  
  // Check if the count of lucky digits is itself a lucky number.
  if (count == 4 || count == 7)
    cout << "YES";
  else
    cout << "NO";
    
  return 0;
}
