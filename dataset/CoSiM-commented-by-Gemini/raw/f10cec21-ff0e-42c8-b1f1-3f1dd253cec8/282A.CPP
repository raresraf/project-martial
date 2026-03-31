/**
 * @file 282A.CPP
 * @brief Simulates a series of increment and decrement operations on a single variable.
 *
 * This program reads an integer `n` followed by `n` statements. Each statement
 * is a string that represents either an increment ("++X", "X++") or a decrement
 * ("--X", "X--") operation on a variable, initialized to zero. The program
 * calculates the final value of the variable after all operations.
 *
 * @author f10cec21-ff0e-42c8-b1f1-3f1dd253cec8
 *
 * @algorithm Simple Simulation
 * @complexity
 * Time: O(N), where N is the number of statements, as we iterate through each statement once.
 * Space: O(1), as storage is constant regardless of the number of statements.
 */

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

// Type Aliases for convenience
typedef unsigned long long ULL;
typedef long long LL;

// Macro for a for-loop from 0 to n-1
#define REP(i, n) FOR(i, 0, n)
// Macro for a generic for-loop
#define FOR(i, a, b) for (int i = a; i < b; i++)
// Macro for a reverse for-loop
#define ROF(i, a, b) for (int i = a; i > b; i--)
/**
 * @def GI
 * @brief Macro to get an integer from standard input.
 */
#define GI                                                                     
  ({                                                                           
    int t;                                                                     
    scanf("%d", &t);                                                           
    t;                                                                         
  })
/**
 * @def GL
 * @brief Macro to get a long long from standard input using cin.
 */
#define GL                                                                     
  ({                                                                           
    LL t;                                                                      
    cin >> t;                                                                  
    t;                                                                         
  })
/**
 * @def GD
 * @brief Macro to get a double from standard input.
 */
#define GD                                                                     
  ({                                                                           
    double t;                                                                  
    scanf("%lf", &t);                                                          
    t;                                                                         
  })
// Shortcut for vector::push_back
#define pb push_back
// Shortcut for std::make_pair
#define mp make_pair
// Define a common modulo for operations
#define MOD 1000000007
// Define a large integer value for infinity
#define INF (int)1e8
// Define a small double value for float comparisons
#define EPS 1e-9
// Macro to iterate over a container
#define TR(a, it)                                                              
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)

int main() {
  // Read the total number of operations.
  long long n = GL;
  // Initialize the counter which represents the variable's value.
  int count = 0;

  /**
   * @brief Main loop to process each operation.
   * Pre-condition: `count` holds the value from previous operations.
   * Invariant: `count` is updated according to the statement read in the current iteration.
   */
  FOR(i, 0, n) {
    string s;
    cin >> s;
    /**
     * @brief Checks for an increment operation.
     * The string represents an increment if it contains a '+' character.
     * Given the problem constraints, the '+' can be at the beginning (e.g., "++X")
     * or at the end (e.g., "X++"). Checking index 0 and 2 covers both cases.
     */
    if (s[0] == '+' || s[2] == '+')
      count++;
    else // Otherwise, it's a decrement operation.
      count--;
  };

  // Print the final value of the counter.
  cout << count;
  return 0;
}
