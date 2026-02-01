/**
 * @file 266A.CPP
 * @brief Solution for a competitive programming problem (Codeforces 266A)
 *        to count the minimum number of stones to remove to make all
 *        adjacent stones different colors. This is equivalent to counting
 *        consecutive identical characters in a string.
 *
 * Algorithm: Linear scan to count adjacent identical characters.
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ for storing the input string, or $O(1)$ if input is processed character by character.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h> // Includes common C++ standard library headers.

using namespace std;

/** @brief Alias for long long integer type. */
typedef long long LL;
/** @brief Alias for vector of integers. */
typedef vector<int> vi;
/** @brief Alias for vector of long long integers. */
typedef vector<long long> vl;

/** @brief Modulo constant, commonly used in competitive programming for modular arithmetic. */
#define MOD 1000000007
/** @brief Loop macro for 0 to n-1. */
#define _for(i, n) for (int i = 0; i < n; i++)
/** @brief Loop macro for 1 to n. */
#define __for(i, n) for (int i = 1; i <= n; i++)
/** @brief Loop macro for a range [s, n]. */
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
/** @brief Loop macro for 0 to n-1 (alternative to _for). */
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
/** @brief Loop macro for a reverse range [e, s]. */
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
/** @brief Iterator loop macro for containers. */
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
/** @brief Shorthand for vector::push_back. */
#define pb push_back
/** @brief Shorthand for std::make_pair. */
#define mp make_pair
/** @brief Shorthand for pair's first element. */
#define fi first
/** @brief Shorthand for pair's second element. */
#define se second
/** @brief Shorthand for getting begin and end iterators of a container. */
#define ALL(x) x.begin(), x.end()

/**
 * @brief Main function to solve the problem of counting consecutive identical stones.
 * Functional Utility: Reads the length of a string and the string itself,
 *                     then iterates through the string to count pairs of
 *                     adjacent identical characters. This count directly
 *                     corresponds to the minimum stones to remove.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares an integer to store the length of the string.
  scanf("%d", &n); // Reads the length of the string from standard input.
  string str; // Declares a string to store the sequence of stones (characters).
  cin >> str; // Reads the string of stones.
  int count = 0; // Initializes a counter for consecutive identical stones.
  /**
   * @brief Block Logic: Iterates through the string to find adjacent identical characters.
   * Invariant: `i` is the current index being checked, and `str.at(i)` is compared with `str.at(i-1)`.
   * Precondition: The loop starts from the second character (index 1) to compare with the preceding one.
   * Postcondition: `count` stores the total number of adjacent identical characters found.
   */
  for (int i = 1; i < n; i++) {
    /**
     * @brief Block Logic: Checks if the current character is identical to the previous one.
     * Precondition: `str.at(i)` and `str.at(i-1)` are valid characters within the string.
     * Postcondition: If identical, `count` is incremented.
     */
    if (str.at(i - 1) == str.at(i))
      count++; // Increments count if consecutive characters are the same.
  }
  printf("%d\n", count); // Prints the final count of adjacent identical stones.
  return 0; // Indicates successful program termination.
}
