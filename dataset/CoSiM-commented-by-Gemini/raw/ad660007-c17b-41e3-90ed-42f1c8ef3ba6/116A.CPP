/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem simulating passenger flow or resource management.
 * This program calculates the maximum number of people simultaneously present in a tram (or items in a resource).
 * It processes a series of entries (`a`) and exits (`b`), maintaining a current count (`c`) and tracking
 * the historical maximum (`s`).
 *
 * Algorithm: Simulation and maximum tracking.
 * Time Complexity: O(N) where N is the number of entry/exit events.
 * Space Complexity: O(1) for storing counter and maximum variables.
 */
#include <bits/stdc++.h>
using namespace std;

#define ll long long int /**< @brief Shorthand for `long long int`. */
#define mp make_pair /**< @brief Shorthand for `std::make_pair`. */
#define pb push_back /**< @brief Shorthand for `std::vector::push_back`. */
#define fi first /**< @brief Shorthand for accessing the first element of a pair. */
#define si second /**< @brief Shorthand for accessing the second element of a pair. */
/**
 * @brief Macro to speed up C++ I/O operations.
 * Disables synchronization with C's stdio and unties cin from cout,
 * which is a common optimization in competitive programming.
 */
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
/**
 * @brief Macro for convenient input of a 1D array.
 * Iterates from `0` to `n-1` to read `n` elements into array `a`.
 * @param a The array to read into.
 * @param n The number of elements to read.
 */
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for convenient input of a 1D array (1-indexed).
 * Iterates from `1` to `n` to read `n` elements into array `a`.
 * @param a The array to read into.
 * @param n The number of elements to read.
 */
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for convenient output of a 1D array.
 * Iterates from `0` to `n-1` to print `n` elements of array `a`,
 * separated by spaces, followed by a newline.
 * @param a The array to print.
 * @param n The number of elements to print.
 */
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for convenient output of a 1D array (1-indexed).
 * Iterates from `1` to `n` to print `n` elements of array `a`,
 * separated by spaces, followed by a newline.
 * @param a The array to print.
 * @param n The number of elements to print.
 */
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for convenient input of a 2D array.
 * Iterates `n` rows and `m` columns to read elements into 2D array `a`.
 * @param a The 2D array to read into.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for convenient input of a 2D array (1-indexed).
 * Iterates `n` rows (from 1 to n) and `m` columns (from 1 to m) to read elements into 2D array `a`.
 * @param a The 2D array to read into.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for convenient output of a 2D array.
 * Iterates `n` rows and `m` columns to print elements of 2D array `a`,
 * separated by spaces, with each row followed by a newline. An extra newline is added at the end.
 * @param a The 2D array to print.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for convenient output of a 2D array (1-indexed).
 * Iterates `n` rows (from 1 to n) and `m` columns (from 1 to m) to print elements of 2D array `a`,
 * separated by spaces, with each row followed by a newline. An extra newline is added at the end.
 * @param a The 2D array to print.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define AS 200001 /**< @brief Represents a common array size or limit, often used in competitive programming. */
#define mod 1000000007 /**< @brief A modulo value, commonly used in competitive programming for large number operations. */
int a, b, s, c; /**< @brief `a`: items entering, `b`: items leaving, `s`: maximum count, `c`: current count. */
/**
 * @brief Main function of the program.
 * Initializes the fast I/O, then reads an initial value into `a` (which is unused as `cin >> a >> b` immediately overwrites `a`).
 * It then enters a loop to read pairs of integers (`a`, `b`) representing entries and exits.
 * The current count `c` is updated, and the maximum count `s` is tracked.
 * Finally, the maximum count `s` is printed.
 */
int main() {
  fastIO // Functional Utility: Applies fast I/O optimizations.

          cin >>
      a; // Reads an initial value for `a`, which is immediately overwritten in the loop.
  /**
   * @brief Loop to process entry and exit events.
   * Reads pairs of integers `a` (exits) and `b` (entries) until end of input.
   * `c` accumulates the current number of people/items.
   * `s` tracks the maximum value `c` has reached.
   * Invariant: `c` always represents the current count, and `s` the maximum count observed so far.
   */
  while (cin >> a >> b) {
    c += b - a; // Update current count: add entries, subtract exits.
    /**
     * @brief Update the maximum count.
     * If the current count `c` exceeds the recorded maximum `s`, `s` is updated.
     */
    if (c > s)
      s = c;
  }
  cout << s; // Prints the maximum number of people/items observed.
}
