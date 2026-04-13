/**
 * @file 282A.CPP
 * @brief This program simulates a basic programming language with a single integer variable 'value'
 * and two operations: increment ('++') and decrement ('--').
 * The operations are read as strings, and the program determines whether to increment or decrement
 * 'value' based on the characters at the first or third position of the operation string.
 * Algorithm: Iterates through a given number of operation strings, applying increment/decrement
 * to a counter based on the operation identified.
 * Time Complexity: O(N * L) where N is the number of operations and L is the average length of an operation string.
 * Space Complexity: O(L) for storing the current operation string.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.

using namespace std; // Brings all names from the std namespace into the current scope.

const double pi = acos(-1.0); // Defines the mathematical constant PI, though not used in this solution.

// Type definitions for competitive programming convenience.
typedef pair<int, int> PII;       // Alias for pair of integers.
typedef vector<int> VI;           // Alias for vector of integers.
typedef vector<string> VS;        // Alias for vector of strings.
typedef vector<PII> VII;          // Alias for vector of pairs of integers.
typedef vector<VI> VVI;           // Alias for vector of vectors of integers.
typedef map<int, int> MPII;       // Alias for map with integer keys and integer values.
typedef set<int> SETI;            // Alias for set of integers.
typedef multiset<int> MSETI;      // Alias for multiset of integers.
typedef long int int32;           // Alias for long int.
typedef unsigned long int uint32; // Alias for unsigned long int.
typedef long long int int64;      // Alias for long long int.
typedef unsigned long long int uint64; // Alias for unsigned long long int.

int main() {
  // Disables synchronization between C-style I/O and C++ streams, and unties cin from cout.
  // This optimizes input/output operations for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int num; // Stores the number of operation statements to process.
  // Reads the number of operation statements.
  cin >> num;
  int value = 0; // The integer variable that will be incremented or decremented.

  /**
   * Block Logic: Processes 'num' operation statements.
   * For each statement, it reads the operation string and updates 'value' accordingly.
   * Invariant: 'value' holds the cumulative result of all processed operations.
   */
  for (int i = 0; i < num; i++) {
    string op; // Stores the current operation statement (e.g., "X++", "++X", "X--", "--X").
    // Reads the operation statement.
    cin >> op;
    /**
     * Conditional Logic: Determines whether to increment or decrement 'value'.
     * It checks if the character at index 0 or index 2 of the operation string is '+'.
     * If either is '+', 'value' is incremented; otherwise, it's decremented.
     * This covers all standard "Bit++" operation formats.
     */
    if (op[0] == '+' || op[2] == '+')
      value++; // Increment 'value'.
    else
      value--; // Decrement 'value'.
  }

  // Prints the final value of 'value' after all operations are processed.
  cout << value;

  // Standard program exit.
  return 0;
}