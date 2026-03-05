/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem.
 *
 * This program calculates the minimum possible capacity of a tram based on the number
 * of passengers exiting and entering at each stop. It simulates the change in
 * passenger count and keeps track of the maximum count observed.
 *
 * @algorithm
 * 1. Read `n`, the total number of stops.
 * 2. Initialize `sum` to 0. This variable will track the number of passengers
 *    currently inside the tram.
 * 3. Initialize `mx` to 0. This variable will store the maximum value that `sum`
 *    reaches, which corresponds to the minimum required capacity.
 * 4. Loop `n` times, once for each stop.
 *    a. Read `a`, the number of passengers exiting, and `b`, the number of
 *       passengers entering.
 *    b. Update the current passenger count: `sum = sum - a + b`.
 *    c. After updating the count for the current stop, compare it with the
 *       maximum seen so far and update `mx` if the current count is greater:
 *       `mx = max(mx, sum)`.
 * 5. After the loop has processed all stops, `mx` holds the minimum capacity
 *    required for the entire trip. Print `mx`.
 *
 * @implementation_note
 * The original code has a special case for the first stop (`i == 0`). While this
 * works, a more general solution can be achieved by initializing `sum = 0` and `mx = 0`
 * before the loop and applying the same logic (`sum = sum - a + b; mx = max(mx, sum);`)
 * for all stops, as the problem statement implies that 0 passengers exit at the first stop.
 *
 * @complexity
 * Time Complexity: O(n), as it iterates through the `n` stops once.
 * Space Complexity: O(1).
 */

#include <iostream>
#include <algorithm> // For std::max

using namespace std;

int main() {
  // n: number of stops
  // a: passengers exiting, b: passengers entering
  // i: loop counter
  // mx: maximum passengers on the tram at any point (minimum capacity)
  // sum: current number of passengers on the tram
  int n, a, b, i, mx, sum;

  cin >> n;

  // Initialize passenger count and max capacity before starting.
  sum = 0;
  mx = 0;

  /**
   * @brief Loop through each of the `n` stops.
   */
  for (i = 0; i < n; i++) {
    cin >> a >> b;

    // This block from the original code handles the first stop as a special case.
    // if (i == 0) {
    //   sum = b;
    //   mx = b;
    //   continue;
    // }

    // Update the current number of passengers.
    // `sum` decreases by `a` and increases by `b`.
    sum -= a;
    sum += b;

    // Update the maximum capacity required if the current number of passengers is higher.
    mx = max(mx, sum);
  }

  // Print the final calculated minimum capacity.
  cout << mx;

  return 0;
}
