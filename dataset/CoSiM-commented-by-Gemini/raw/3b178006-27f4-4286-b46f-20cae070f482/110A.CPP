/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 *
 * @details A "nearly lucky number" is a number where the count of its "lucky
 * digits" (4 and 7) is itself a lucky number. This program checks if a given
 * number adheres to this definition.
 *
 * @algorithm The program first counts the number of lucky digits (4 or 7) in the
 * input number `n`. This is done by iterating through the digits of the number
 * using modulo and division operations. After the loop, it checks if the final
 * count is equal to 4 or 7. If it is, the number is nearly lucky, and "YES" is
 * printed; otherwise, "NO" is printed.
 *
 * @complexity
 * Time Complexity: O(log10(N)), where N is the input number, as the work is
 *                  proportional to the number of digits in N.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
int main() {
  ll n;
  cin >> n;
  ll k = n, r, c = 0; // k: copy of n, r: remainder/digit, c: lucky digit count

  /**
   * @brief Block Logic: Iterates through each digit of the number to count lucky digits.
   * @invariant `c` holds the count of lucky digits in the portion of the number
   * already processed. `k` holds the remaining part of the number.
   */
  while (k > 0) {
    r = k % 10;
    if (r == 4 || r == 7)
      c++;
    k /= 10;
  }
  
  // Check if the count of lucky digits is itself a lucky number (4 or 7).
  if (c == 4 || c == 7)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}