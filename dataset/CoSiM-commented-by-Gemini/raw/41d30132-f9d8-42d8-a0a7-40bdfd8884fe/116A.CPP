/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *
 * @algorithm
 * The problem is to calculate the minimum required capacity of a tram, which is
 * equivalent to finding the maximum number of passengers inside the tram at any
 * single point in time.
 *
 * This implementation first reads all the input data into arrays and then
 * processes it in a separate loop.
 *
 * The algorithm is as follows:
 * 1. Read the total number of stops, `n`.
 * 2. Create two arrays, `a` and `b`, to store the number of exiting and
 *    entering passengers for each stop, respectively.
 * 3. Read all `n` pairs of exit/enter values into these arrays.
 * 4. Initialize `maxim` (maximum capacity) and `cap` (current passengers) to 0.
 * 5. Loop `n` times, simulating each stop using the data from the arrays.
 * 6. In each iteration, update `cap` by subtracting the exiting passengers `a[i]`
 *    and adding the entering passengers `b[i]`.
 * 7. Use `std::max` to update `maxim` if the current `cap` is greater.
 * 8. After the loop, print the final `maxim` value.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stops, due to the two
 *   separate loops that each run N times.
 * - Space Complexity: O(N), due to storing all `n` exit and enter values in
 *   arrays `a` and `b`.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, a[1001], b[1001], maxim = 0, cap = 0;
  cin >> n;
  /**
   * @brief Reads all exit (a[i]) and enter (b[i]) passenger counts for all
   * stops into arrays before starting the simulation.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i] >> b[i];

  /**
   * @brief Simulates the tram's journey stop by stop to find the peak passenger count.
   */
  for (int i = 0; i < n; i++) {
    // Update the number of people currently on the tram.
    cap = b[i] - a[i] + cap;
    // Keep track of the maximum number of people seen so far.
    maxim = max(maxim, cap);
  }
  cout << maxim;
}