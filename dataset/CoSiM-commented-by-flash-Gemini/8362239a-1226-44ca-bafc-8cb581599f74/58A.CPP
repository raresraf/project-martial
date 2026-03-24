/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" competitive programming problem.
 *
 * This program determines if the string "hello" can be formed as a subsequence
 * from a given input string. It uses a straightforward iterative approach to match
 * characters of "hello" in their correct order within the input string.
 *
 * Algorithm: Subsequence matching (greedy).
 * Time Complexity: O(L_s) where L_s is the length of the input string `s`, as it
 *                  involves a single pass over the string.
 * Space Complexity: O(L_h) where L_h is the length of the target subsequence "hello" (constant 5).
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

// Functional Utility: Custom macro for a forward for-loop.
#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
// Functional Utility: Custom macro for a backward for-loop.
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
// Functional Utility: Custom macro for iterating through a container.
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
// Functional Utility: Custom macro for getting begin and end iterators of a container.
#define _all(v) v.begin(), v.end()
// Functional Utility: Custom macro for initializing memory with zeros.
#define __(v) memset(v, 0, sizeof(v))

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

typedef long long LL;     ///< Alias for long long integer type.
typedef unsigned long long ULL; ///< Alias for unsigned long long integer type.

/**
 * @brief Overloads the `operator+=` to provide a convenient way to push elements into a vector.
 * @tparam T The type of elements in the vector.
 * @param v Reference to the vector.
 * @param x The element to push back.
 * @return vector<T>& A reference to the modified vector.
 */
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief Contains the core logic for solving the "Chat room" problem.
 *
 * Reads an input string `s` and checks if "hello" is a subsequence within it.
 * Prints "YES" if found, "NO" otherwise.
 */
void solve() {
  string s;
  /**
   * @brief Reads the input string from standard input.
   *
   * Pre-condition: A string is available on standard input.
   * Post-condition: `s` stores the input string.
   */
  cin >> s;
  string hello = "hello"; // Target subsequence.
  int index = 0;          // Index for tracking characters in "hello".
  int count = 0;          // Unused variable, can be removed.

  /**
   * @brief Iterates through the input string `s` to find "hello" as a subsequence.
   *
   * Invariant: `index` progresses only when a matching character from `hello` is found in `s`.
   * Pre-condition: `s` is initialized with the input string.
   * Post-condition: `index` indicates how many characters of "hello" were found in `s` sequentially.
   * Functional Utility: Performs a greedy match of "hello" within `s`.
   */
  _for(i, 0, s.length()) {
    // Conditional Logic: Checks if the current character in `s` matches the character
    //                    at `index` in "hello".
    if (s[i] == hello[index])
      index++; // Advances the index in "hello" if a match is found.
  }

  /**
   * @brief Determines and prints the result based on whether "hello" was fully matched.
   *
   * Conditional Logic: If `index` reaches 5 (length of "hello"), then "hello" is a subsequence.
   * Functional Utility: Outputs the final answer ("YES" or "NO") to standard output.
   */
  if (index == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

/**
 * @brief Main entry point of the program.
 *
 * Sets up fast I/O and calls the `solve()` function.
 */
int main() {
  /**
   * @brief Optimizes C++ standard streams for faster input and output.
   * Functional Utility: Disables synchronization with C's stdio and unties cin from cout,
   *                     which is crucial for performance in competitive programming.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);
// Conditional Compilation: Enables I/O redirection for debugging purposes if HIEUNV macro is defined.
#ifdef HIEUNV
  freopen("input.txt", "r", stdin); // Redirects standard input from "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif
  // Redundant I/O optimization calls, can be removed as they are already set above.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  solve(); // Calls the function containing the problem's solution logic.
}