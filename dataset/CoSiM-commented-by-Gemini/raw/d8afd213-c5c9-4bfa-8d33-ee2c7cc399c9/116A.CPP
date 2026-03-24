/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * @details This program calculates the minimum capacity required for a tram by tracking the
 * maximum number of passengers on board at any given time.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the total number of tram stops, `n`.
 * 2. Initialize `inTram` to 0 to track the current number of passengers.
 * 3. Initialize `max` to 0. This variable will store the peak passenger count.
 * 4. Loop `n` times, once for each stop:
 *    a. Read the number of passengers exiting (`ai`) and entering (`bi`).
 *    b. Update the current passenger count by adding the net change (`-ai + bi`).
 *    c. If the new `inTram` count is greater than `max`, update `max` to the new peak value.
 * 5. After the loop, `max` holds the minimum capacity required for the tram. Print this value.
 *
 * @complexity
 * Time Complexity: O(n), where `n` is the number of stops, due to the single loop.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  int ai, bi;
  // `inTram` tracks the current number of passengers. `max` tracks the peak number of passengers.
  int inTram = 0, max = 0;
  cin >> n; // Read the number of stops.
  /**
   * Block Logic: This loop simulates the tram's journey, updating the passenger
   * count at each stop to find the peak occupancy.
   */
  while (n--) {
    cin >> ai >> bi; // ai: passengers exiting, bi: passengers entering.
    
    // Update the number of people currently on the tram.
    inTram += (-ai + bi);
    
    /**
     * Block Logic: Check if the current passenger count is the highest so far.
     * If it is, this becomes the new required minimum capacity.
     */
    if (inTram > max)
      max = inTram;
  }

  // Print the maximum number of passengers seen, which is the minimum required capacity.
  cout << max << endl;

  return 0;
}