/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem using a direct calculation of Manhattan distance.
 * @details The problem requires finding the minimum number of moves (row or column swaps)
 * to move the single '1' in a 5x5 matrix to the center at (3,3) (using 1-based indexing).
 * This is equivalent to calculating the Manhattan distance between the '1's position and the center.
 * This implementation finds the coordinates of the '1' and then directly computes this distance.
 *
 * @note This file contains a number of helper functions (isprime, gcd, exp) and macros
 * that are part of a user's competitive programming template and are not used in this specific solution.
 *
 * @verdict OK
 * @submission_id 19585704
 */

#include <bits/stdc++.h>
typedef long long ll;
typedef unsigned long long ull;
#define pi 3.141592653589793238462643383279502884197169399375105820974944592307816406286
#define mod 1000000007
using namespace std;

// --- Template function, not used in this solution. ---
bool isprime(ull n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (ull i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}

// --- Template function, not used in this solution. ---
long long int gcd(long long int a, long long int b) {
  if (b == 0)
    return a;
  else
    return gcd(b, a % b);
}

// --- Template function, not used in this solution. ---
long long int exp(long long int value, long long int power) {
  long long int result = 1;
  while (power > 0) {
    if (power & 1LL) {
      result *= value;
      result %= mod;
    }
    value *= value;
    value %= mod;
    power /= 2LL;
  }
  return result;
}

int main() {
  // Fast I/O setup.
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  
  // A large number of variables declared as part of a template, most are unused.
  long long int f, cnt, len, p, q, r, t, i, j, k, l, n, m, x, y, z, b, c, s;
  
  /**
   * @brief Reads the matrix and finds the 1-based coordinates of the '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> k;
      if (k == 1) {
        x = i + 1; // Store 1-based row index.
        y = j + 1; // Store 1-based column index.
      }
    }
  }
  
  // Directly calculate and print the Manhattan distance to the center (3,3).
  cout << abs(x - 3) + abs(y - 3);

  return 0;
}