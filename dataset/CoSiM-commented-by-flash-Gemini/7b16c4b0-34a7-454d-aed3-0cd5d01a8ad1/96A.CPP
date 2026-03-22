/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 *        This program analyzes a sequence of player positions (represented by '0's and '1's)
 *        to determine if a "dangerous situation" exists. A dangerous situation is defined as
 *        seven or more players of the same team standing side by side.
 *
 * Algorithm:
 *   1. Reads a string `s` representing the sequence of players.
 *   2. If the string length is less than 7, it's impossible to have a dangerous situation, so "NO" is printed.
 *   3. Otherwise, it iterates through all possible starting positions for a 7-character subsequence.
 *   4. For each 7-character subsequence, it checks if all characters within it are identical.
 *   5. If an identical 7-character subsequence is found, "YES" is printed, and the program terminates.
 *   6. If no such subsequence is found after checking all possibilities, "NO" is printed.
 *
 * Time Complexity: O(L * K), where L is the length of the input string `s` and K is the required
 *                  length of consecutive identical characters (7 in this problem). Since K is constant,
 *                  this simplifies to O(L).
 * Space Complexity: O(L) for storing the input string `s`, plus O(1) for auxiliary variables.
 */

#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#define mod 1000000007 ///< Constant for modulo operations, typically used in problems involving large numbers.
#define Debug(x) cout << '>' << #x << ':' << x << endl; ///< Debug macro to print variable name and value.
#define FOR(i, a, b) for (long long int i = (a); i < (b); i++) ///< Macro for a for-loop with a specific range.
#define REP(i, a) for (long long int i = (0); i < (a); i++) ///< Macro for a for-loop starting from 0.
const int INF = 1 << 29; ///< Constant representing infinity, used for initializations in algorithms.
typedef long long int ll; ///< Typedef for long long int.
typedef unsigned long long int ull; ///< Typedef for unsigned long long int.

// Utility function to compute the greatest common divisor (GCD). Not used in main.
ll gcd(ll a, ll b) { return (a % b == 0) ? abs(b) : gcd(b, a % b); }
// Utility function to compute the least common multiple (LCM). Not used in main.
ll lcm(ll a, ll b) { return (a * b) / gcd(a, b); }
// Utility function to compute the maximum of two numbers. Not used in main.
ll max(ll a, ll b) {
  if (a > b)
    return a;
  return b;
}
// Utility function to compute power (x^y). Not used in main.
ll power(ll x, ll y) {
  ll res = 1;
  while (y > 0) {
    if (y & 1)
      res = res * x;
    y = y >> 1;
    x = x * x;
  }
  return res;
}
// Utility function to compute power (x^y) modulo 'mod'. Not used in main.
ll powermod(ll x, ll y) {
  ll res = 1;
  x = x % mod;
  while (y > 0) {
    if (y & 1)
      res = (res * x) % mod;
    y = y >> 1;
    x = (x * x) % mod;
  }
  return res;
}
// Utility function to check if a number is prime. Not used in main.
bool isPrime(ll n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (int i = 5; i * i <= n; i += 6) {
    if ((n % i == 0) || (n % (i + 2) == 0))
      return false;
  }
  return true;
}
// Utility function to compute nCr (combinations). Not used in main.
ll nCr(ll n, ll r) {
  if (r == 0)
    return 1;
  return (n * nCr(n - 1, r - 1)) / r;
}
// Utility function to count divisors of a number. Not used in main.
ll countDivisors(ll n) {
  ll cnt = 0;
  for (int i = 1; i <= sqrt(n); i++) {
    if (n % i == 0) {
      if (n / i == i)
        cnt++;
      else
        cnt = cnt + 2;
    }
  }
  return cnt;
}
// Utility function to compute modulo with positive result. Not used in main.
ll modulo(ll a, ll b) {
  ll r = a % b;
  return r < 0 ? r + b : r;
}
// Utility function for binary search on an array. Not used in main.
ll bsearch_array(ll n, ll *arr, ll first, ll last) {
  if (first == last) {
    if (arr[first] == n)
      return first + 1;
    else
      return -1;
  } else {
    ll m = (first + last) / 2;
    if (arr[m] == n)
      return m + 1;
    else {
      if (arr[m] < n)
        return bsearch_array(n, arr, m + 1, last);
      else
        return bsearch_array(n, arr, first, m);
    }
  }
}
// Utility function for binary search on a vector. Not used in main.
ll bsearch_vect(std::vector<ll> &nums, int target) {

  if (nums.size() == 0)
    return -1;

  int left = 0, right = nums.size() - 1;
  while (left < right) {
    ll mid = left + (right - left) / 2;
    if (nums[mid] == target) {
      return mid;
    } else if (nums[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Football" problem.
   *
   * Reads a string representing player positions and determines if there are
   * 7 or more consecutive players of the same team. Optimizes I/O operations.
   *
   * @return int Returns 0 upon successful execution.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s;
  cin >> s;

  /**
   * Conditional Logic: Checks if the string is too short to contain a dangerous situation.
   * Pre-condition: `s` has been read.
   * Post-condition: If `s.length() < 7`, prints "NO" and exits.
   */
  if (s.length() < 7) {
    cout << "NO" << endl;
    return 0;
  }

  /**
   * Block Logic: Iterates through the string `s` using a sliding window of size 7.
   * Invariant: `i` is the starting index of the current 7-character window.
   * Pre-condition: `s.length()` is at least 7.
   */
  REP(i, s.length() - 6) {
    bool flag = false; // Flag to indicate if characters in the current window are different.
    /**
     * Block Logic: Checks if all characters within the current 7-character window are identical.
     * Invariant: `j` iterates from `i+1` to `i+6` (inclusive), comparing `s[j]` with `s[i]`.
     * Pre-condition: `s[i]` is the first character of the current window.
     */
    FOR(j, i + 1, i + 7) {
      // Conditional Logic: If a character in the window differs from the first character,
      // it's not a dangerous sequence.
      if (s[i] != s[j]) {
        flag = true;
        break; // Exit inner loop as this window is not dangerous.
      }
    }

    /**
     * Conditional Logic: If `flag` is still false, it means all 7 characters were identical.
     * Pre-condition: The inner loop has completed for the current window.
     * Post-condition: Prints "YES" and terminates if a dangerous situation is found.
     */
    if (!flag) {
      cout << "YES" << endl;
      return 0; // Found a dangerous situation, exit.
    }
  }
  // Final output if no dangerous situation was found after checking all windows.
  cout << "NO" << endl;
  return 0;
}