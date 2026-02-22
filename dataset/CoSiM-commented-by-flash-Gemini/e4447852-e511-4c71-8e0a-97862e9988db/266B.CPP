/**
 * @file 266B.CPP
 * @brief Solution for Codeforces Problem 266B: Queue at the School.
 * @details This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each unit of time (second), if a boy 'B' is immediately in front of a girl 'G', they swap places.
 * This process is simulated for 't' seconds. The implementation handles the "simultaneous" nature
 * of swaps by iterating the queue from right to left and skipping a position after a swap.
 * @algorithm Iterative simulation with string manipulation.
 * Time Complexity: $O(T \cdot N)$ where T is the number of time steps and N is the length of the queue.
 * Space Complexity: $O(N)$ to store the queue string.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace std;

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
#define mod 1000000007
#define s(a) scanf("%d", &a)
#define sll(a) scanf("%lld", &a)
#define ss(a) scanf("%s", a)
#define ll long long
#define mp make_pair
#define pb push_back
#define fi first
#define se second

char a[55];

/**

 * @file 266B.CPP

 * @brief Solution for Codeforces Problem 266B: Queue at the School.

 * @details This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').

 * In each unit of time (second), if a boy 'B' is immediately in front of a girl 'G', they swap places.

 * This process is simulated for 't' seconds. The implementation handles the "simultaneous" nature

 * of swaps by iterating the queue from right to left and skipping a position after a swap.

 * @algorithm Iterative simulation with string manipulation.

 * Time Complexity: $O(T \cdot N)$ where T is the number of time steps and N is the length of the queue.

 * Space Complexity: $O(N)$ to store the queue string.

 */

#include <algorithm> // Required for std::swap.

#include <cmath>     // Generally useful, not strictly needed for this problem.

#include <cstdio>    // Required for scanf and printf.

#include <cstdlib>   // Generally useful, not strictly needed for this problem.

#include <cstring>   // Required for string manipulation, though char array is used.

#include <deque>     // Generally useful, not strictly needed for this problem.

#include <iostream>  // Generally useful, not strictly needed for this problem (scanf/printf preferred here).

#include <map>       // Generally useful, not strictly needed for this problem.

#include <queue>     // Generally useful, not strictly needed for this problem.

#include <set>       // Generally useful, not strictly needed for this problem.

#include <stack>     // Generally useful, not strictly needed for this problem.

#include <string>    // Generally useful, not strictly needed for this problem (char array preferred here).

#include <utility>   // Generally useful, not strictly needed for this problem.

#include <vector>    // Generally useful, not strictly needed for this problem.



using namespace std; // Uses the standard namespace.



typedef pair<int, int> pii; // @typedef pii: Alias for pair of integers.

typedef vector<int> vi;    // @typedef vi: Alias for vector of integers.

typedef vector<vi> vvi;    // @typedef vvi: Alias for vector of vectors of integers.

typedef vector<pii> vii;   // @typedef vii: Alias for vector of pairs of integers.



// Macro definitions for common competitive programming tasks.

#define mod 1000000007 // @def mod: Modulo value, often used in modular arithmetic.

#define s(a) scanf("%d", &a) // @def s: Shorthand for reading an integer.

#define sll(a) scanf("%lld", &a) // @def sll: Shorthand for reading a long long integer.

#define ss(a) scanf("%s", a) // @def ss: Shorthand for reading a string (char array).

#define ll long long         // @def ll: Alias for long long.

#define mp make_pair         // @def mp: Shorthand for std::make_pair.

#define pb push_back         // @def pb: Shorthand for std::vector::push_back.

#define fi first             // @def fi: Shorthand for accessing the first element of a pair.

#define se second            // @def se: Shorthand for accessing the second element of a pair.



char a[55]; // @var a: Character array to store the queue of boys and girls. Max size 54 characters + null terminator.



int main() {

  /**

   * @brief Main function: orchestrates the queue simulation logic.

   */

  int n, // @var n: The number of people in the queue (length of the queue string).

      t; // @var t: The total number of seconds (time steps) to simulate.



  s(n), s(t);    // Reads the queue length 'n' and simulation time 't'.

  scanf("%s", a); // Reads the initial queue configuration into character array `a`.



  /**

   * @brief Simulates the queue reordering over 't' seconds.

   * @block_logic Outer loop iterates for each second of the simulation.

   * @invariant The queue state evolves after each second according to the swap rules.

   */

  for (int i = 0; i < t; ++i) {

    /**

     * @brief Performs simultaneous swaps of 'B' and 'G' pairs in the queue for the current second.

     * @block_logic Inner loop iterates the queue from right to left (`j = n-1` down to `1`).

     * @invariant `j` represents the current position of the girl in a potential 'BG' pair.

     * @details Iterating from right to left and decrementing `j` after a swap ensures that a boy

     *          who just moved forward (to `j-1`) does not immediately swap again in the same

     *          second with the girl now behind him, simulating simultaneous swaps.

     */

    for (int j = n - 1; j > 0; --j) {

      if (a[j] == 'G' && a[j - 1] == 'B') { // Checks if a 'B' is in front of a 'G'.

        swap(a[j], a[j - 1]);               // Swaps the 'G' and 'B' characters.

        j--;                                // Skips the new position of 'G' to prevent re-evaluation in the same second.

      }

    }

  }



  // Functional Utility: Prints the final configuration of the queue after 't' seconds.

  printf("%s\n", a);



  return 0; // Indicates successful execution.

}