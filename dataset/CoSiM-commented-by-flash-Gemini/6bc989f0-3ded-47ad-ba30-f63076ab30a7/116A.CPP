/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem: "Tram".
 * Simulates passengers entering and exiting a tram to find the maximum number of passengers in the tram at any point.
 * Algorithm: Iterative simulation, tracking current passenger count and maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1)
 */
#include <cstdio>

using namespace std;

/**
 * @brief Main function to solve the Tram problem.
 * Reads the number of stops 'N', then for each stop, reads the number of passengers exiting 'a' and entering 'b'.
 * Calculates the current number of passengers and tracks the maximum number of passengers encountered.
 */
int main() {
  int N;
  int a, b, curr = 0, max = 0;
  fscanf(stdin, "%d", &N);

  /**
   * Block Logic: Iterates through each tram stop.
   * Pre-condition: 'N' stops are to be processed. 'curr' and 'max' are initialized to 0.
   * Invariant: At the start of each iteration, 'curr' holds the passenger count from the previous stop, and 'max' holds the highest count seen so far.
   */
  for (int i = 0; i < N; ++i) {
    fscanf(stdin, "%d%d", &a, &b);

    /**
     * Block Logic: Updates the current number of passengers in the tram.
     * 'a' passengers exit, and 'b' passengers enter.
     */
    curr = (curr - a) + b;
    /**
     * Block Logic: Updates the maximum number of passengers if the current count exceeds it.
     * This tracks the peak capacity requirement of the tram.
     */
    if (curr > max) {
      max = curr;
    }
  }
  fprintf(stdout, "%d", max);

  return 0;
}
