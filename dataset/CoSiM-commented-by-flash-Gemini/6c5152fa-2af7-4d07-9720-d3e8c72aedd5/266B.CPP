/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B: "Queue at the School".
 * Simulates a queue where boys ('B') and girls ('G') swap positions if a boy is
 * immediately followed by a girl. This process is repeated for a given number of
 * time steps.
 * Algorithm: Iterative simulation with a linear scan for swaps.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the
 * length of the queue.
 * Space Complexity: O(N) for storing the queue.
 */
#include <cstdio>     // For standard input/output operations (e.g., scanf, printf)
#include <iostream>   // For standard input/output streams (e.g., cin, cout)
#include <cstring>    // For string manipulation functions (e.g., memset)
#include <sstream>    // For string stream manipulation
#include <string>     // For string class

#include <deque>      // For double-ended queue container
#include <list>       // For list container
#include <map>        // For map container
#include <queue>      // For queue container
#include <set>        // For set container
#include <stack>      // For stack container
#include <vector>     // For vector container

#include <algorithm>  // For algorithms like sort, min, max
#include <bitset>     // For bitset manipulation
#include <functional> // For function objects
#include <iomanip>    // For input/output manipulators (e.g., setprecision)
#include <numeric>    // For numeric operations (e.g., accumulate)
#include <utility>    // For utility components (e.g., pair)

#include <cctype>     // For character handling functions (e.g., isalpha, tolower)
#include <cmath>      // For mathematical functions (e.g., sqrt, pow)
#include <cstdlib>    // For general utilities (e.g., abs, atoi)
#include <ctime>      // For time and date functions
using namespace std;

// Global variable for case numbering in competitive programming, though not used here.
int caseno = 1;
#define NL '\n' // Macro for newline character, often used for performance in competitive programming.
// Macros for common loop patterns.
#define FOR(I, J, K) for (I = J; I < K; I++)
#define REV(I, J, K) for (I = J; I > K; I--)
#define SF scanf   // Macro for scanf
#define PF printf  // Macro for printf
#define CLR(ar) memset(ar, 0, sizeof(ar)) // Macro to clear an array with zeros.
#define SET(ar) memset(ar, -1, sizeof(ar)) // Macro to set an array with -1.
#define PC() printf("Case %d: ", caseno++) // Macro for printing case number.
#define READ() freopen("in.txt", "r", stdin) // Macro for redirecting input from a file.
#define WRITE() freopen("out.txt", "w", stdout) // Macro for redirecting output to a file.
#define BOOST std::ios_base::sync_with_stdio(0); // Macro to speed up C++ I/O.

// Type definitions for convenience, common in competitive programming.
typedef long long LL;              // Represents a 64-bit signed integer.
typedef unsigned long long ULL;    // Represents a 64-bit unsigned integer.
typedef long long int64;           // Alias for long long.
typedef unsigned long long uint64; // Alias for unsigned long long.

// Constants for common values.
#define INF 2147483647      // Represents infinity for 32-bit integers.
#define MOD 1000000007      // A common modulo value in competitive programming.
const double PI = 2 * acos(0.0); // Mathematical constant PI.
const double EPS = 1e-11;     // Epsilon for floating-point comparisons.
const int SIZE = 1e6;         // A common size for arrays or containers.

string S; // Global string variable to store the queue.

/**
 * @brief Main function to solve the Queue at the School problem.
 * Reads the length of the queue (N) and the number of time steps (T).
 * Simulates the swapping process of 'B' and 'G' characters in the queue for T steps.
 */
int main() {
  BOOST // Optimize C++ standard streams for faster input/output.
  int tcases, I, J, K, N, n, m, cnt = 0, len, t; // Declare integer variables.
  /// READ();  // Commented out: Typically used for file input.
  /// WRITE(); // Commented out: Typically used for file output.
  cin >> len >> t; // Read the length of the queue and the number of time steps.
  cin >> S;        // Read the initial queue configuration.

  /**
   * Block Logic: Simulates the queue transformation over 't' time steps.
   * Pre-condition: 'S' contains the initial queue, 'len' is its length, 't' is the number of steps.
   * Invariant: After each iteration, 'cnt' increments, representing one time step completed.
   */
  while (cnt < t) {
    /**
     * Block Logic: Performs a single pass through the queue to swap 'B' and 'G' pairs.
     * Pre-condition: Current state of 'S'.
     * Invariant: In one pass, each eligible 'BG' pair is swapped to 'GB'.
     */
    for (I = 0; I < len - 1;) {
      /**
       * Block Logic: Checks for a "BG" pair and performs a swap.
       * If S[I] is 'B' and S[I+1] is 'G', they are swapped.
       * The loop index 'I' is advanced by 2 to skip the newly swapped 'GB' pair,
       * ensuring that the same 'B' does not swap multiple times in a single second.
       * Otherwise, 'I' is advanced by 1 to check the next character.
       */
      if (S[I] == 'B' && S[I + 1] == 'G') {
        S[I + 1] = 'B';
        S[I] = 'G';
        I += 2;
      } else
        I++;
    }
    cnt++; // Increment the time step counter.
  }
  cout << S << NL; // Output the final state of the queue.

  return 0; // Indicate successful execution.
}
