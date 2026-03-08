/**
 * @file 236A.CPP
 * @brief Determines if a username corresponds to a "boy" or "girl" based on the parity of distinct characters.
 *
 * Problem Name: Boy or Girl (Codeforces 236A)
 * Algorithm: Uses a `std::set` to efficiently count distinct characters in the input string.
 *            The parity of this count determines the output ("CHAT WITH HER!" for even, "IGNORE HIM!" for odd).
 * Time Complexity: O(L log D) where L is the length of the string and D is the number of distinct characters.
 *                  Given D <= 26 (English alphabet), this is effectively O(L).
 * Space Complexity: O(D) for storing distinct characters in the set, and O(L) for the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, especially <iostream>, <string>, <set>.
using namespace std;

// Functional Utility: Defines 'll' as a shorthand for 'long long int'.
#define ll long long int
// Functional Utility: Defines 'mp' as a shorthand for 'make_pair'.
#define mp make_pair
// Functional Utility: Defines 'pb' as a shorthand for 'push_back'.
#define pb push_back
// Functional Utility: Defines 'fi' as a shorthand for 'first' (e.g., in pairs).
#define fi first
// Functional Utility: Defines 'si' as a shorthand for 'second' (e.g., in pairs).
#define si second
// Functional Utility: Optimizes I/O operations by decoupling C++ streams from C stdio and untying cin from cout.
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
// Functional Utility: Macros for simplified 1D array input (0-indexed).
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
// Functional Utility: Macros for simplified 1D array input (1-indexed).
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
// Functional Utility: Macros for simplified 1D array output (0-indexed).
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
// Functional Utility: Macros for simplified 2D array input (0-indexed).
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
// Functional Utility: Macros for simplified 2D array input (1-indexed).
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
// Functional Utility: Macros for simplified 2D array output (0-indexed).
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
// Functional Utility: Macros for simplified 2D array output (1-indexed).
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
// Functional Utility: Constant for array size (e.g., max N).
#define AS 200001
// Functional Utility: Constant for modulo operation.
#define mod 1000000007

/**
 * @brief Main function to determine "Boy or Girl" based on distinct characters in a username.
 * @pre Input consists of a single string (username).
 * @post Prints "CHAT WITH HER!" if the number of distinct characters is even, otherwise "IGNORE HIM!".
 */
int main() {
  fastIO; // Functional Utility: Calls the I/O optimization macro.

  string s; // Variable to store the input username.
  // Functional Utility: Reads the username string from standard input.
  cin >> s;

  // Functional Utility: Creates a set of characters from the input string.
  // A `std::set` automatically stores only unique elements.
  set<char> set_name(begin(s), end(s));

  /**
   * Block Logic: Checks the parity of the number of distinct characters.
   * Pre-condition: `set_name.size()` holds the count of unique characters.
   * Invariant: If the count is even, it's a "girl"; if odd, it's a "boy".
   */
  if (set_name.size() % 2 == 0) { // Checks if the number of distinct characters is even.
    cout << "CHAT WITH HER!";     // Output for a "girl".
  } else {
    cout << "IGNORE HIM!"; // Output for a "boy".
  }
  // No explicit return 0; is needed for main in C++ if it reaches the end.
  // However, for competitive programming, it's often good practice to include it.
}