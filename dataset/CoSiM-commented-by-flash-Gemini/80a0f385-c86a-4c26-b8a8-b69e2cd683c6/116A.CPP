/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit (`a`) and then a certain number enter (`b`). The program efficiently tracks the
 * current number of passengers (`curr`) and the maximum number of passengers observed (`highest`)
 * without storing all intermediate states in an array.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h>
using namespace std;
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit (`a`) and then a certain number enter (`b`). The program efficiently tracks the
 * current number of passengers (`curr`) and the maximum number of passengers observed (`highest`)
 * without storing all intermediate states in an array.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream for cin/cout, algorithm for max).
using namespace std;     // Uses the standard namespace.

// Define useful macro aliases for competitive programming.
#define length(a) (int)a.size()                                     // @def length: Returns the size of a container as an int.
#define all(a) a.begin(), a.end()                                   // @def all: Returns iterators to the beginning and end of a container.
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower); // @def tolowercase: Converts a string to lowercase.
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper); // @def touppercase: Converts a string to uppercase.
#define lsb(x) (x & (-x))                                           // @def lsb: Returns the least significant bit of `x`.
#define ctoi(a) (int)a - 48                                         // @def ctoi: Converts a character digit to an integer.
#define itoc(a) a + '0'                                             // @def itoc: Converts an integer digit to a character.
#define MOD 1000000007                                              // @def MOD: A common modulo value for arithmetic operations.
#define ll long long                                                // @def ll: Alias for long long integer.

int main() {
  /**
   * @brief Main function: orchestrates the tram simulation logic.
   */

  // Functional Utility: Optimizes C++ standard streams for competitive programming.
  // Disables synchronization with C's stdio and unties cin from cout.
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int n; // @var n: The total number of tram stops.
  cin >> n; // Reads the total number of tram stops.

  int highest = 0; // @var highest: Stores the maximum number of passengers encountered in the tram.
  int curr = 0;    // @var curr: Stores the current number of passengers in the tram.

  /**
   * @brief Simulates the tram journey through each stop, updating passenger counts and tracking the maximum.
   * @block_logic Iterates `n` times, processing passenger changes at each stop.
   * @invariant `curr` reflects the current passenger count, and `highest` holds the peak passenger count.
   */
  for (int i = 0; i < n; i++) {
    int a, b; // @var a: Number of passengers exiting at the current stop. @var b: Number of passengers entering at the current stop.
    cin >> a >> b; // Reads the number of exiting and entering passengers for the current stop.
    curr -= a;     // Passengers exit.
    curr += b;     // Passengers enter.
    highest = max(highest, curr); // Updates `highest` if the current `curr` is higher.
  }

  // Functional Utility: Prints the minimum capacity required (maximum number of passengers observed).
  cout << highest << "\n";
  return 0; // Indicates successful execution.
}