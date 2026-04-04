/**
 * @file 116A.CPP
 * @brief Calculates the maximum capacity required for a tram.
 *
 * This program simulates a tram journey with 'n' stops. At each stop, it records
 * the number of passengers who exit and the number who enter. The objective is
 * to determine the maximum number of passengers present on the tram at any point
 * throughout its journey.
 *
 * Algorithm: Iterative simulation. Keep a running total of passengers and update
 *            the maximum observed value in each step.
 * Time Complexity: O(N), where N is the number of stops, as each stop is processed once.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function: Simulates tram journey to find maximum passenger count.
   * @return 0 if the program executes successfully.
   */
  int n = 0, a = 0, b = 0, peopleIn = 0, peopleOut = 0, mAx = 0, total = 0,
      res = 0; // Initialize variables: 'n' for stops, 'a' for exited, 'b' for entered,
               // 'peopleIn', 'peopleOut' for current stop, 'mAx' for change in people,
               // 'total' for current people on tram, 'res' for max people on tram.
  // n the number of stops
  cin >> n; // Read the number of stops.
  /**
   * @brief Loop through each stop of the tram journey.
   * @pre 'n' is a positive integer.
   * @post 'res' holds the maximum number of people on the tram throughout the journey.
   * @invariant 'total' accurately reflects the number of people on the tram after each stop.
   */
  for (int i = 0; i < n; i++) {
    // a is the number of the passengers leave
    // b is the number of the passengers enter
    cin >> a; // Read the number of passengers leaving the tram at the current stop.
    cin >> b; // Read the number of passengers entering the tram at the current stop.
    peopleOut = a; // Assign 'a' to 'peopleOut' (redundant if 'a' and 'b' are directly used).
    peopleIn = b;  // Assign 'b' to 'peopleIn' (redundant if 'a' and 'b' are directly used).
    mAx = peopleIn - peopleOut; // Calculate the net change in passengers at this stop.
    total = total + mAx;        // Update the total number of people currently on the tram.
    /**
     * @brief Conditional block to update the maximum number of people observed on the tram.
     * @pre 'total' reflects the current number of passengers.
     */
    if (total > res) {
      res = total; // Update 'res' if the current total is a new maximum.
    }
  }
  cout << res; // Output the maximum number of people observed on the tram.

  return 0;
}
