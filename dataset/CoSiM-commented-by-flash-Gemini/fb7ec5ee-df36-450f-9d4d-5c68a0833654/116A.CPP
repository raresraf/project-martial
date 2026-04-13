/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem: "Tram".
 *
 * This program calculates the maximum number of people observed inside a tram
 * at any single point in time, given the number of people exiting and entering
 * at each stop.
 *
 * Algorithm: Iterative simulation of tram occupancy.
 * Time Complexity: O(N), where N is the number of tram stops. Each stop involves a constant number of arithmetic operations.
 * Space Complexity: O(1), as only a few variables are used to track current and maximum occupancy.
 */
#include <algorithm> // Required for std::max.
#include <cmath>     // Potentially for math functions, though not used in solve.
#include <cstdlib>   // Potentially for general utilities, though not used in solve.
#include <deque>     // Potentially for deque container, though not used in solve.
#include <fstream>   // Potentially for file operations, though not used in solve.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <limits>    // Potentially for numeric limits, though not used in solve.
#include <list>      // Potentially for list container, though not used in solve.
#include <map>       // Potentially for map container, though not used in solve.
#include <queue>     // Potentially for queue container, though not used in solve.
#include <set>       // Potentially for set container, though not used in solve.
#include <sstream>   // Potentially for string stream manipulation, though not used in solve.
#include <stack>     // Potentially for stack container, though not used in solve.
#include <string>    // Required for std::string.
#include <utility>   // Potentially for std::pair, though not used explicitly in solve.
#include <vector>    // Required for std::vector.

// using
/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// typedef
typedef vector<int> VI;   ///< @brief Alias for a vector of integers.
typedef vector<VI> VII;   ///< @brief Alias for a 2D vector of integers.
typedef vector<string> VS; ///< @brief Alias for a vector of strings.
typedef vector<long> VL;   ///< @brief Alias for a vector of long integers.
typedef long long LL;     ///< @brief Alias for long long integer.
typedef vector<LL> VLL;   ///< @brief Alias for a vector of long long integers.

// container utils
#define PB push_back ///< @brief Appends an element to a std::vector.
#define PF push_front ///< @brief Prepends an element to a std::deque.
#define GRT(x) greater<(x)>() ///< @brief Returns a greater comparator for type x.
#define ASORT(x) sort((x).begin(), (x).end()) ///< @brief Sorts a container in ascending order.
#define DSORT(x, y) sort((x).begin(), (x).end(), greater<(y)>()) ///< @brief Sorts a container in descending order using a custom comparator.
#define FILL(x, y) fill((x).begin(), (x).end(), (y)) ///< @brief Fills a container with a specified value.
// Macro to copy elements from one container to another, clearing the destination first.
#define COPY(x, y)                                                             \
  (y).clear();                                                                 \
  copy((x).begin(), (x).end(), back_inserter(y))

// repetition (looping macros)
#define FOR(i, a, b) for (int i = (a); i < (b); i++) ///< @brief Loop from 'a' (inclusive) to 'b' (exclusive) with integer 'i'.
#define FORL(i, a, b) for (long i = (a); i < (b); i++) ///< @brief Loop from 'a' (inclusive) to 'b' (exclusive) with long 'i'.
#define REP(i, n) FOR(i, 0, n) ///< @brief Loop from 0 (inclusive) to 'n' (exclusive) with integer 'i'.
#define REPL(i, n) FORL(i, 0, n) ///< @brief Loop from 0 (inclusive) to 'n' (exclusive) with long 'i'.

// debug macros (for printing variable values to stderr)
#define DUMP(a) cerr << #a << " = " << (a) << endl ///< @brief Prints a single variable's name and value.
#define DUMP2(a, b)                                                            \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << endl ///< @brief Prints two variables' names and values.
#define DUMP3(a, b, c)                                                         \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << ", " << #c       \
       << " = " << (c) << endl ///< @brief Prints three variables' names and values.

int n; // Global variable to store the number of tram stops.

/**
 * @brief Solves the "Tram" problem by simulating passenger flow.
 *
 * This function calculates the maximum number of people observed inside a tram
 * at any single point in time, given the number of people exiting and entering
 * at each stop.
 * Algorithm: Iterative simulation of tram occupancy.
 * Time Complexity: O(N), where N is the number of tram stops, as it iterates through
 *                  each stop once.
 * Space Complexity: O(1), as it only uses a few variables to track current and maximum occupancy.
 */
void solve() {
  cin >> n; // Reads the total number of tram stops.
  long a, b; // 'a' stores the number of people exiting, 'b' stores the number of people entering.
  long p = 0; // 'p' tracks the current number of people in the tram, initialized to 0.
  long p_max = 0; // 'p_max' stores the maximum number of people observed, initialized to 0.

  /**
   * @brief Loop simulating passenger activity at each tram stop.
   * Block Logic: For each of the 'n' stops, it reads the number of people exiting and entering,
   * updates the current passenger count, and then updates the maximum passenger count if
   * the current count exceeds the previously recorded maximum.
   * Pre-condition: 'n' is the total number of stops, 'p' is current occupancy, 'p_max' is max occupancy so far.
   * Invariant: At the end of each iteration, 'p' reflects the current occupancy and 'p_max' reflects
   * the highest occupancy encountered up to that stop.
   */
  REP(i, n) {
    cin >> a >> b; // Reads 'a' (exiting passengers) and 'b' (entering passengers) for the current stop.
    p -= a;        // Decrements current passengers by those exiting.
    p += b;        // Increments current passengers by those entering.
    p_max = max(p, p_max); // Updates 'p_max' to the larger of the current 'p' and previous 'p_max'.
  }

  cout << p_max << endl; // Prints the overall maximum number of people observed in the tram.
}

/**
 * @brief Main function: program entry point.
 * Calls the `solve()` function to execute the core logic.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return 0 upon successful execution.
 */
int main(int argc, char *argv[]) {
  solve(); // Executes the tram occupancy simulation.

  return 0; // Indicates successful program termination.
}
