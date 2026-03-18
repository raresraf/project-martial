/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem, a variation of "Soldier and Bananas".
 * Calculates the total cost of 'd' items where the cost increases arithmetically (n*k, 2*n*k, ..., d*n*k).
 * Then determines how much additional money is needed given an initial amount 'k'.
 * Algorithm: Direct calculation using arithmetic series sum, followed by a comparison.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
/*
   By Anmol Gupta :)
*/
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;
#pragma GCC optimize("O3") // Compiler optimization flag for GCC.
#define pb push_back       // Macro for std::vector::push_back.
#define debug(x) cout << #x << " " << x << endl; // Debug macro to print variable name and value.
#define mp make_pair       // Macro for std::make_pair.
#define ff first           // Macro for accessing the first element of a pair.
#define ss second          // Macro for accessing the second element of a pair.
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end()); // Removes duplicate elements from a sorted vector.
#define endl "\n"          // Macro for newline character, often used for performance.
#define MOD 1000000007     // Common modulo constant.
#define MOD9 1000000009    // Another common modulo constant.
#define gc getchar_unlocked // Faster character input.
#define ms(s, n) memset(s, n, sizeof(s)) // Sets memory with a byte value.
#define prec(n) fixed << setprecision(n) // Sets floating-point output precision.
#define eps 0.000001       // Epsilon for floating-point comparisons.
#define pll pair<ll, ll>   // Type alias for pair of long longs.
#define all(v) v.begin(), v.end() // Returns iterators to the beginning and end of a container.
#define bolt ios::sync_with_stdio(0) // Disables synchronization with C stdio for faster I/O.
#define forr(i, n) for (ll i = 0; i < n; i++) // Macro for a common for loop pattern.
#define eof (scanf("%d", &n)) != EOF // Checks for end of file after reading an integer.
typedef long long ll; // Type alias for long long.

/**
 * @brief Main function to solve the competitive programming problem.
 * Reads 'n', 'k', and 'd' (representing initial cost increment, initial money, and number of items).
 * Calculates the total cost and prints the amount of money to borrow, or 0 if none is needed.
 */
int main() {
  ios_base::sync_with_stdio(false); // Untie C++ streams from C standard streams.
  cin.tie(NULL); // Untie cin from cout.
  cout.tie(NULL); // Untie cout.
  // printf("%I64d", n) // Example comment from original code, likely for 64-bit integer printing.
  ll n, k, d, tot; // Declare variables: 'n' initial increment, 'k' initial money, 'd' number of items, 'tot' total cost.
  cin >> n >> k >> d; // Read input values.

  /**
   * Block Logic: Calculate the total cost of 'd' items.
   * The cost follows an arithmetic progression: n*d, 2*n*d, ..., d*n*d.
   * The sum of the first 'd' terms of an arithmetic series where the first term is `n*d` and common difference is `n*d`
   * is `(d * (first_term + last_term)) / 2 = (d * (n*d + d*n*d)) / 2`
   * Simplifying, `(d * n * d * (1 + d)) / 2 = n * d * (d + 1) * d / 2`.
   * Given the variable usage, it seems to be `k * (1+2+...+w) = k * w * (w+1)/2`.
   * So this line computes (cost of 1st banana * sum of first 'd' integers).
   */
  tot = n * d * (d + 1); // Equivalent to `k * w * (w+1)` in Soldier and Bananas.
  tot = tot / 2;         // Completes the sum: `k * w * (w+1) / 2`.
  /**
   * Block Logic: Determine the amount of money to borrow.
   * If the total cost ('tot') exceeds the available money ('k'), the difference is borrowed.
   * Otherwise, no money needs to be borrowed.
   */
  if (tot > k)
    cout << tot - k; // Output the deficit.
  else
    cout << "0"; // Output 0 if no deficit.

  return 0; // Indicate successful execution.
}