/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem (Codeforces 116A).
 * @author competitive-programmer
 *
 * @details This program calculates the minimum possible capacity of a tram.
 * It is given the number of passengers exiting and entering at each stop.
 * The minimum capacity must be equal to the maximum number of passengers
 * inside the tram at any point in time.
 *
 * Algorithm:
 * 1. Initialize `maxCount` (the required capacity) and `count` (current
 *    passengers) to 0.
 * 2. Read the number of tram stops, `t`.
 * 3. Loop `t` times to simulate the journey from stop to stop.
 * 4. In each stop:
 *    a. Read the number of exiting passengers (`a`) and entering passengers (`b`).
 *    b. Update the current passenger count: `count = count - a + b`.
 *    c. Compare the new `count` with `maxCount`. If `count` is greater,
 *       update `maxCount` to this new peak value.
 * 5. After iterating through all the stops, `maxCount` will hold the maximum
 *    number of passengers that were ever on the tram simultaneously. Print this value.
 *
 * Time Complexity: O(t), where `t` is the number of stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // `t` is the number of tram stops.
  int t;
  cin >> t;
  
  // `maxCount` stores the peak number of passengers found so far.
  // `count` stores the current number of passengers on the tram.
  int maxCount = 0, count = 0;
  
  /**
   * Block Logic: This loop simulates the tram's journey, processing one stop
   * per iteration.
   */
  while (t--) {
    // `a` is the number of exiting passengers, `b` is the number of entering passengers.
    int a, b;
    cin >> a >> b;
    
    // Update the current number of passengers.
    count -= a;
    count += b;
    
    // If the current passenger count exceeds the maximum recorded so far, update the max.
    if (count > maxCount) {
      maxCount = count;
    }
  }
  
  // The minimum capacity required is the maximum number of passengers ever on the tram.
  cout << maxCount << endl;
}
