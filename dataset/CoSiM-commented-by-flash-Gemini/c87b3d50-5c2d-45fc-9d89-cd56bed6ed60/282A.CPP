/**
 * @file 282A.CPP
 * @brief This program simulates a basic programming language with a single integer variable 'y'
 * and two operations: increment ('++') and decrement ('--').
 * The operations can be represented in four forms: "X++", "++X", "X--", and "--X".
 * Algorithm: Parses operation strings and applies increment/decrement to a counter variable.
 * Time Complexity: O(N * L) where N is the number of statements and L is the average length of an operation string.
 * Space Complexity: O(N * L) for storing the array of operation strings.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
#include <string>        // Required for string manipulation.
#define ll long long     // Defines a macro 'll' as a shorthand for 'long long'.
using namespace std;
int main() {

// Conditional compilation block for local debugging.
// If not compiled in an online judge environment, it redirects standard input from "input.txt"
// and standard output to "output.txt".
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int x; // Stores the number of statements (operations) to process.
  int y = 0; // The integer variable that will be incremented or decremented.
  // Reads the number of statements.
  cin >> x;
  string c[x]; // Declares a dynamic array of strings to store each operation statement.
  /**
   * Block Logic: Iterates 'x' times to process each operation statement.
   * For each statement, it reads the operation and updates 'y' accordingly.
   * Invariant: 'y' holds the cumulative result of all processed operations.
   */
  for (int i = 0; i < x; i++) {
    // Reads the current operation statement.
    cin >> c[i];

    /**
     * Conditional Logic: Checks the specific form of the operation string to determine
     * whether to increment or decrement 'y'.
     */
    if (c[i] == "X++") {
      y++;
    } else if (c[i] == "++X") {
      ++y;
    } else if (c[i] == "X--") {
      y--;
    } else if (c[i] == "--X") {
      --y;
    }
  }
  // Prints the final value of 'y' after all operations are processed.
  cout << y << endl;
  // Standard program exit.
  return 0;
}