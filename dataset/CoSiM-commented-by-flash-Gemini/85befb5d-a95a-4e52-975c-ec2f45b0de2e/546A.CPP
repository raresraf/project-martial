/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 * This program calculates the amount of money a soldier needs to borrow to buy a
 * certain number of bananas, given the cost pattern and the soldier's initial money.
 * The first banana costs `k`, the second `2k`, and so on.
 *
 * Algorithm: Arithmetic series sum calculation. The total cost of 'w' bananas
 * is `k * (1 + 2 + ... + w)`. This sum is then compared with the soldier's
 * initial money to determine the borrowed amount.
 * Time Complexity: O(1) as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 */

// @brief Standard C++ library includes and competitive programming macros.
#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).
#define nl '\n'          // @brief Macro for newline character.
#define _time_ clock_t tStart = clock(); // @brief Macro to start timer for execution time.
#define _show_                                                                 \
  printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC); // @brief Macro to show execution time.
#define ll long long int // @brief Type definition for long long int.
#define loop(a, b) for (int i = a; i <= b; ++i) // @brief Macro for a for loop.
#define count_1(n) __builtin_popcountll(n) // @brief Macro to count set bits in a long long.
#define pb push_back // @brief Macro for vector push_back.
#define F first      // @brief Macro for pair first element.
#define S second     // @brief Macro for pair second element.
#define mp make_pair // @brief Macro for pair make_pair.
#define clr(x) x.clear() // @brief Macro to clear a container.
#define MOD 1000000007 // @brief Modulo constant.
#define itoc(c) ((char)(((int)'0') + c)) // @brief Macro to convert int to char.
#define vi vector<int> // @brief Type definition for vector of ints.
#define pll pair<ll, ll> // @brief Type definition for pair of long longs.
#define pii pair<int, int> // @brief Type definition for pair of ints.
#define all(p) p.begin(), p.end() // @brief Macro for iterating over a container.
#define mid(s, e) (s + (e - s) / 2) // @brief Macro for calculating midpoint.
// @brief Macro for a common competitive programming loop structure (reading test cases).
#define sv()                                                                   \
  ll t, n;                                                                     \
  scanf("%lld", &t);                                                           \
  n = t;                                                                       \
  while (t--)
// @brief Macro for another common competitive programming loop structure (reading test cases with cin).
#define tcase()                                                                \
  ll t, n;                                                                     \
  cin >> t;                                                                    \
  n = t;                                                                       \
  while (t--)
#define iscn(num) scanf("%d", &num); // @brief Macro for reading an integer with scanf.

using namespace std;
void FAST_IO();
int main() {
  /**
   * @brief Main function to solve the Soldier and Bananas problem.
   * Reads the cost of the first banana (k), initial money (n), and number of bananas (w).
   * Calculates the total cost and determines the amount to borrow.
   */
  //    _time_ // @brief Macro to start timing execution (commented out).
  // FAST_IO(); // @brief Function call to optimize I/O (commented out by default).
  ll k, n, w; // @brief k: cost of first banana, n: initial money, w: number of bananas.
  cin >> k >> n >> w;
  
  // Block Logic: Calculate the total cost of 'w' bananas.
  // The cost follows an arithmetic progression: k, 2k, ..., wk.
  // The sum of an arithmetic series is `count * (first + last) / 2`.
  // Here, `count = w`, `first = k`, `last = w*k`.
  // So, sum = `w * (k + wk) / 2 = w * k * (1 + w) / 2`.
  // The code uses `w++` effectively making it `(w+1)*(w)/2`, which is a common sum for 1 to w.
  w++; // Inline: Effectively changes 'w' to 'w+1' for calculation convenience, then used as original 'w'.
  ll temp = (w * (w - 1)) / 2; // Inline: Calculates sum of 1 to (w-1), which is 1 to original w.
  temp = temp * k - n; // Inline: Total cost (sum * k) minus initial money.
  
  // Block Logic: Determine the amount to borrow.
  // If the calculated amount is negative (soldier has enough money), then 0 needs to be borrowed.
  if (temp < 0)
    temp = 0;
  
  // Block Logic: Output the amount to be borrowed.
  cout << temp << endl;
  //   _show_ // @brief Macro to show execution time (commented out).
  return 0; // @brief Indicates successful program termination.
}

/**
 * @brief Optimizes C++ standard input/output operations.
 * Unties cin/cout from C standard streams and disables synchronization with C stdio.
 * Also includes commented-out sections for file I/O redirection for offline testing.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0); // @brief Disables synchronization with C standard I/O.
  cin.tie(0);                   // @brief Unties cin from cout.
  cout.tie(0);                  // @brief Unties cout from cin.
  // cout.setf(ios::fixed); // @brief Sets fixed-point notation for floating-point output (commented out).
  // cout.precision(20);    // @brief Sets precision for floating-point output (commented out).
#ifndef _offline // @brief Conditional compilation for offline mode.
  freopen("input.txt", "r", stdin); // @brief Redirects standard input from "input.txt" (commented out).
  // freopen("output.txt", "w", stdout); // @brief Redirects standard output to "output.txt" (commented out).
#endif
}
