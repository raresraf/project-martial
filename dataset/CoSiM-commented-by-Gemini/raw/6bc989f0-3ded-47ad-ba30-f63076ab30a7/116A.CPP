
#include <cstdio>

using namespace std;

/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A, "Tram".
 * @details The program calculates the minimum required capacity of a tram by
 * determining the maximum number of passengers on board at any single time.
 *
 * Algorithm:
 * The program iterates through each of the `N` tram stops. At each stop, it
 * reads the number of passengers exiting (`a`) and entering (`b`). It maintains
 * a running count of the current number of passengers (`curr`). This count is
 * updated at each stop by subtracting the exiting passengers and adding the
 * entering ones. A separate variable (`max`) keeps track of the highest value
 * `curr` has reached. After the loop, `max` holds the peak passenger count,
 * which corresponds to the minimum required capacity.
 *
 * Time Complexity: O(N), where N is the number of stops, as it involves a
 * single pass through all the stops.
 * Space Complexity: O(1), as it only uses a few variables to store the current
 * state, not the entire list of entries/exits.
 */
int main() {
  // N: the number of tram stops.
  int N;
  // a: passengers exiting, b: passengers entering at the current stop.
  // curr: current number of passengers on the tram.
  // max: maximum number of passengers seen so far.
  int a, b, curr = 0, max = 0;

  // Using C-style input to read the number of stops.
  fscanf(stdin, "%d", &N);

  /**
   * @brief Simulate the journey through all N stops.
   *
   * Invariant: `curr` holds the number of passengers on the tram just before
   * the `i`-th stop, and `max` holds the maximum number of passengers seen
   * up to that point.
   */
  for (int i = 0; i < N; ++i) {
    fscanf(stdin, "%d%d", &a, &b);

    // Update the current passenger count after the exchange at this stop.
    curr = (curr - a) + b;
    
    // If the current count is a new maximum, update max.
    if (curr > max) {
      max = curr;
    }
  }
  
  // Using C-style output to print the result.
  fprintf(stdout, "%d", max);

  return 0;
}
