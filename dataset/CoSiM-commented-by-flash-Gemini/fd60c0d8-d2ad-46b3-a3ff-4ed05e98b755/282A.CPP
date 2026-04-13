/**
 * @file 282A.CPP
 * @brief This program simulates a basic programming language with a single integer variable 'ans'
 * and two operations: increment ('++') and decrement ('--').
 * The operations are read as strings, and the program determines whether to increment or decrement
 * 'ans' based on the character at the second position of the operation string.
 * Algorithm: Parses operation strings, checks the second character for '+' or '-', and applies
 * the corresponding arithmetic operation to 'ans'.
 * Time Complexity: O(N * L) where N is the number of statements and L is the average length of an operation string.
 * Space Complexity: O(L) for storing the temporary operation string.
 */

#pragma comment(linker, "/STACK:16777216") // Linker directive, primarily for MSVC, increases stack size.

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.

using namespace std; // Brings all names from the std namespace into the current scope.

// Macros for competitive programming convenience.
#define ms(ar, a) memset(ar, a, sizeof(ar))             // Shorthand for memset, sets memory to a value.
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++) // Shorthand for a forward for loop.
#define db(x) cout << (#x) << " = " << x << endl;      // Debug macro to print variable name and value.
#define pb push_back                                   // Shorthand for std::vector::push_back.
#define mp make_pair                                   // Shorthand for std::make_pair.
#define X first                                        // Shorthand for accessing the first element of a pair.
#define Y second                                       // Shorthand for accessing the second element of a pair.

// Type definitions for convenience.
typedef long long ll;         // Alias for long long integer.
typedef pair<int, int> pii;   // Alias for pair of integers.

/**
 * @brief Template function to convert any type to a string.
 * @tparam _T The type of the variable to convert.
 * @param a The variable to convert.
 * @return A string representation of the variable.
 */
template <class _T> inline string tostr(const _T &a) {
  ostringstream os("");
  os << a;
  return os.str();
}

int n_global; // Global variable 'n', declared but shadowed by local 'n' in main.

int main() {
  int n_local; // Local variable 'n' to store the number of statements.
  // Reads the number of statements to process using C-style input.
  scanf("%d", &n_local);
  int ans = 0; // The integer variable that will be incremented or decremented.

  /**
   * Block Logic: Processes 'n_local' operation statements.
   * For each statement, it reads the operation string and updates 'ans' accordingly.
   * Invariant: 'ans' holds the cumulative result of all processed operations.
   */
  while (n_local--) {
    string tmp; // Stores the current operation statement.
    // Reads the operation statement (e.g., "X++", "++X", "X--", "--X").
    cin >> tmp;
    /**
     * Conditional Logic: Determines whether to increment or decrement 'ans' based on the
     * character at index 1 of the operation string. If it's '+', increment; otherwise, decrement.
     * This works for formats like "X++", "++X", "X--", "--X" because '+' or '-' will always be at index 1.
     */
    if (tmp[1] == '+')
      ans++; // Increment 'ans'.
    else
      ans--; // Decrement 'ans'.
  }

  // Prints the final value of 'ans' after all operations are processed, followed by a newline.
  printf("%d\n", ans);

  // Standard program exit.
  return 0;
}
