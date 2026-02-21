// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 */

/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (Codeforces 116A) with special handling for the first stop.
 *
 * @details This program calculates the minimum possible capacity of a tram by simulating
 * the number of passengers at each stop and tracking the peak.
 *
 * @algorithm The program simulates passenger flow stop by stop. It uses a `flag`
 * variable to identify and handle the first stop as a special case.
 * 1. At the first stop, the current passenger count (`temp`) and max capacity (`capacity`)
 *    are initialized to the number of entering passengers.
 * 2. At each subsequent stop, the current passenger count is updated by subtracting
 *    exiting passengers and adding entering ones.
 * 3. The max capacity is updated if the current passenger count exceeds the previous max.
 * 4. After processing all stops, the final max capacity is the answer.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * Space Complexity: O(1), as only a few variables are needed for the simulation.
 */
#include <iostream>

using namespace std;

int main() {
  int n, capacity = 0;
  int a, b, temp;

  cin >> n;

  // `flag` is used to identify the first iteration of the while loop.
  int flag = n;

  /**
   * @brief Block Logic: Simulates passenger count over `n` stops.
   * @invariant `temp` holds the current passenger count. `capacity` holds the
   * maximum passenger count observed so far.
   */
  while (n > 0) {
    cin >> a >> b;

    // Special case for the first stop.
    if (n == flag) {
      capacity = b;
      temp = b;
    }

    else {
      // Update passenger count for subsequent stops.
      temp = (temp - a) + b;

      if (temp >= capacity)
        capacity = temp;
    }

    n--;
  }

  cout << capacity << endl;
}
