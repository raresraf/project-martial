/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces (Problem 116A).
 * @details This program calculates the minimum possible capacity of a tram. It does
 * this by simulating the change in passenger numbers over a series of stops and
 * finding the maximum number of passengers inside the tram at any given time.
 *
 * @section
 * Algorithm:
 * 1. Read the number of tram stops, 'n'.
 * 2. Initialize a variable for the current passenger count ('sum') and another for the
 *    maximum passenger count ('max'), both to zero.
 * 3. Loop 'n' times, once for each stop.
 * 4. In each iteration, read the number of exiting ('a') and entering ('b') passengers.
 * 5. Update the current passenger count: sum = sum - a + b.
 * 6. Compare the updated 'sum' with 'max'. If 'sum' is greater, update 'max' to 'sum'.
 * 7. After the loop, 'max' holds the peak passenger count, which is the minimum required capacity.
 * 8. Print the value of 'max'.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(n), where 'n' is the number of stops, due to the single loop through all stops.
 * - Space Complexity: O(1), as only a few integer variables are used regardless of the input size.
 */
#include <iostream>
using namespace std;

int main() {
  int stn;
  cin >> stn;
  // 'sum' tracks the current number of passengers on the tram.
  // 'max' tracks the maximum number of passengers seen so far, which equals the required capacity.
  int a, b, sum = 0, max = 0;

  /**
   * Block Logic: Simulates the tram's journey stop by stop.
   * Invariant: At the start of each iteration, 'sum' is the number of people on the
   * tram as it arrives at the stop, and 'max' is the peak occupancy up to that point.
   */
  while (stn--) {
    // Read the number of passengers exiting (a) and entering (b).
    cin >> a >> b;
    // Update the current number of passengers on the tram after the exchange.
    sum += (b - a);
    // Check if the current passenger count is a new maximum.
    if (max < sum) {
      max = sum;
    }
  }
  // Output the minimum required capacity.
  cout << max;
  return 0;
}