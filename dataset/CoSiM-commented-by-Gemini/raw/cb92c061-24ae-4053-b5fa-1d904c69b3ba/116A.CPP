/**
 * @file 116A.CPP
 * @brief An efficient, on-the-fly solution for Codeforces problem 116A, "Tram".
 * @details This program determines the minimum required capacity for a tram by
 * calculating the peak passenger count over a series of stops. It does this
 * in a memory-efficient way by processing data as it's read.
 *
 * @section Algorithm
 * The solution simulates the passenger count at each stop to find the maximum.
 * 1. The number of stops, `n`, is read.
 * 2. Two variables are initialized to 0: `currently` to track the current number
 *    of passengers, and `total` to track the maximum number of passengers
 *    seen so far (this will be the final required capacity).
 * 3. A loop runs `n` times, once for each stop.
 *    a. The number of exiting (`a`) and entering (`b`) passengers are read.
 *    b. The `currently` count is updated.
 *    c. The `total` (max capacity) is updated if `currently` is now larger.
 * 4. After the loop, `total` contains the minimum required capacity, which is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of stops, due to the single loop.
 * - Space Complexity: O(1), as the program uses a fixed number of variables
 *   regardless of the number of stops.
 */
#include <iostream>
using namespace std;

int main() {
  // `currently`: tracks the number of passengers in the tram at any moment.
  int currently = 0;
  // `total`: tracks the maximum value that `currently` reaches.
  int total = 0;
  // `n`: the number of stops.
  int n;
  cin >> n;
  // `a`: passengers exiting, `b`: passengers entering.
  int a, b;
  /**
   * @brief This loop processes data for each of the `n` stops.
   * It simulates the change in passenger numbers and updates the maximum
   * required capacity on the fly.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    currently -= a;
    currently += b;
    // If the current number of passengers is the highest seen so far,
    // update the required capacity.
    if (currently > total)
      total = currently;
  }
  cout << total << "\n";
}
