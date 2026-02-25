/**
 * @file 753A.CPP
 * @brief This program solves a problem involving partitioning a positive integer 'n' into the maximum possible number of distinct positive integers.
 * It calculates the largest 'm' such that the sum of the first 'm' natural numbers (1 + 2 + ... + m) is less than or equal to 'n'.
 * Then, it prints the first 'm-1' integers and the remaining value of 'n' (which represents the m-th integer), ensuring all 'm' parts are distinct and positive.
 * Algorithm: Mathematical derivation based on sum of arithmetic series and greedy partitioning.
 * Time Complexity: O(sqrt(N)) due to the square root operation and a loop up to 'm', where 'm' is approximately sqrt(2N).
 * Space Complexity: O(1) for storing a few variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
#define endl '\n' // Redefines 'endl' to a newline character for faster output in competitive programming.
#define ll long long int // Defines 'll' as a shorthand for 'long long int'.
#define loop(i, a, b) for (ll i = a; i <= b; ++i) // Macro for a common for-loop pattern.
#define pb push_back // Shorthand for 'push_back' for vectors.
#define F first // Shorthand for accessing the first element of a pair.
#define S second // Shorthand for accessing the second element of a pair.
#define mp make_pair // Shorthand for 'make_pair'.
#define clr(x) x.clear() // Macro to clear a container.
#define MOD 1000000007 // Defines a modulo constant, often used in problems involving large numbers.
#define itoc(c) ((char)(((int)'0') + c)) // Macro to convert an integer to its character representation.
#define vl vector<ll> // Defines 'vl' as a shorthand for a vector of long longs.
#define SZ(x) (x).size() // Macro to get the size of a container.
#define all(p) p.begin(), p.end() // Macro for passing iterators to functions like sort.
#define mid(s, e) (s + (e - s) / 2) // Macro to calculate the middle point safely.
#define sv()                                                                   \
  ll t, n;                                                                     \
  scanf("%lld", &t);                                                           \
  n = t;                                                                       \
  while (t--) // Macro for handling multiple test cases with scanf.
#define tcase()                                                                \
  ll t, n;                                                                     \
  cin >> t;                                                                    \
  n = t;                                                                       \
  while (t--) // Macro for handling multiple test cases with cin.
#define iscn(num) scanf("%d", &num); // Macro for integer input using scanf.
using namespace std;
bool file = 0, rt = 1;
clock_t tStart;
void FAST_IO();
////////////////////////

int main() {

  FAST_IO(); // Calls a function to optimize I/O operations.
  ////////////////////////
  ll n; // Declares a long long integer 'n' to store the input sum.
  cin >> n; // Reads the integer 'n'.
  /**
   * Functional Utility: Calculates the maximum number of distinct positive integers 'm'
   * whose sum is less than or equal to 'n'. This is derived from the arithmetic series sum formula
   * (m * (m + 1) / 2 <= n), solved for 'm' using the quadratic formula.
   * We take the floor of the result since 'm' must be an integer.
   */
  ll m = (sqrt(8 * n + 1) - 1) / 2;
  cout << m << endl; // Prints the maximum number of distinct positive integers 'm'.
  /**
   * Block Logic: Greedily assigns the smallest distinct positive integers (1, 2, ..., m-1)
   * to be part of the sum.
   * Invariant: After each iteration, 'n' is reduced by the value of 'i', and 'i' is printed.
   * This ensures that 'n' eventually holds the value of the largest part of the sum.
   */
  for (int i = 1; i < m; i++)
    cout << i << " ", n -= i;
  /**
   * Functional Utility: Prints the remaining value of 'n'.
   * This remaining 'n' ensures that the sum of all printed integers equals the original input,
   * and due to the greedy selection of smaller numbers, this 'n' will be distinct from the
   * previously printed '1' to 'm-1' (unless 'm' is small, but still distinct from used numbers)
   * and will be the largest part of the sum.
   */
  cout << n << endl;

  ////////////////////////
  // Conditional Logic: Debugging and timing output, typically for local testing in competitive programming.
  if (rt && file) {
    printf("\nTime taken: %.6fs", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  }
  return 0; // Indicates successful execution.
}

/**
 * @brief Optimizes C++ standard I/O streams for competitive programming.
 * Disables synchronization between C-style I/O and C++ streams, and unties cin from cout.
 * This significantly speeds up input and output operations.
 * Also includes conditional file I/O for debugging/local testing.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0); // Unties C++ streams from C standard streams.
  // cin.tie(NULL); // Unties cin from cout, preventing flushes before input operations.
  // cout.tie(NULL); // Unties cout from cout, potentially allowing interleaved output.
  // cout.setf(ios::fixed); // Sets floating-point output format.
  // cout.precision(2); // Sets floating-point precision.
  // Conditional Logic: Initializes a timer if real-time tracking and file I/O are enabled.
  if (rt && file) {
    tStart = clock();
  }
  // Conditional Logic: Redirects standard input and output to files for local testing.
  if (file) {
#ifndef _offline
    freopen("input.txt", "r", stdin); // Redirects stdin to "input.txt".
    freopen("output.txt", "w", stdout); // Redirects stdout to "output.txt".
#endif
  }
}
