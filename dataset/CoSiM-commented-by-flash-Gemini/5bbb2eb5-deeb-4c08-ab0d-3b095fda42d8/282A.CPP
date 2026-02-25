/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces (Problem 282A).
 *        This program simulates operations on a single variable 'count'.
 *        It reads 'n' operation strings. Each string is parsed to determine
 *        if 'count' should be incremented or decremented.
 * Algorithm: Counter Simulation / String Character Checking
 * Time Complexity: O(N * C), where N is the number of operations and C is a small
 *                  constant number of character checks per string. Effectively O(N).
 * Space Complexity: O(L) to store the operation string (L is fixed at 3), effectively O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream and string.
using namespace std;

// Macros for competitive programming convenience.
#define f first   ///< @brief Shorthand for accessing the first element of a pair.
#define s second  ///< @brief Shorthand for accessing the second element of a pair.
#define mp make_pair ///< @brief Shorthand for std::make_pair, used for creating pairs.
#define pb push_back ///< @brief Shorthand for vector::push_back.
typedef long long ll; ///< @brief Defines 'll' as a shorthand for long long int.

/**
 * @brief Main function to process Bit++ operations and determine the final value of 'count'.
 * @return 0 if the program executes successfully.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Commented out: for redirecting input from a file.
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // speeding up I/O operations.
  ios_base::sync_with_stdio(false);
  
  int n; // Declare an integer to store the number of operations.
  // Read the number of operations.
  cin >> n;
  
  int count = 0; // Initialize the counter variable 'count' to 0.
  
  /**
   * Block Logic: Processes 'n' operation strings, modifying 'count' based on character checks.
   * Invariant: 'count' reflects the cumulative result of operations processed so far.
   */
  while (n--) { // Loop 'n' times, once for each operation.
    string xs; // Declare a string to store the current operation.
    // Read the operation string.
    cin >> xs;
    
    /**
     * Block Logic: Checks if the operation is an increment or a decrement.
     * Pre-condition: 'xs' contains an operation string like "X++", "++X", "X--", or "--X".
     * Post-condition: 'count' is either incremented or decremented by 1.
     * The condition checks if either the first character or the third character is a '+'.
     * If true, it's an increment. Otherwise, by elimination, it's a decrement.
     */
    if (xs[0] == '+' || xs[2] == '+')
      ++count; // Increment 'count'.
    else
      --count; // Decrement 'count'.
  }
  // Print the final value of 'count' followed by a newline.
  cout << count << endl;
  return 0; // Indicate successful program termination.
}