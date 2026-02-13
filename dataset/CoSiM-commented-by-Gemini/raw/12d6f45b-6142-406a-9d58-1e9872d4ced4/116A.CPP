/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem (Codeforces 116A).
 * @author competitive-programmer
 *
 * @details This program calculates the minimum required capacity for a tram by
 * simulating the number of passengers at each stop. The capacity must be at
 * least the maximum number of passengers on the tram at any given time.
 *
 * Algorithm:
 * 1. Read the number of stops, `n`.
 * 2. Initialize `cap` (required capacity) and `x` (current passengers) to 0.
 * 3. Loop `n` times, once for each stop.
 * 4. In each stop, read the number of passengers exiting (`a`) and entering (`b`).
 * 5. Update the current passenger count: `x = x - a + b`.
 * 6. If the current count `x` is greater than the max capacity `cap` seen so far,
 *    update `cap` to `x`.
 * 7. After the loop, `cap` will hold the peak passenger count. Print this value.
 *
 * Time Complexity: O(n), where n is the number of stops.
 * Space Complexity: O(1).
 */
// 116A
#include <iostream>
using namespace std;
int main() {
  // `n` is the number of stops.
  int n;
  // `a` and `b` are the number of passengers exiting and entering at a stop.
  int a, b;
  // `cap` will store the maximum capacity required.
  // `x` stores the current number of passengers in the tram.
  int cap = 0;
  cin >> n;
  int x = 0;
  
  /**
   * Block Logic: This loop simulates the tram's journey stop by stop.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the current passenger count.
    x = x - a;
    x = x + b;
    
    // If the current passenger count is a new peak, update the capacity.
    if (x > cap)
      cap = x;
  }
  
  // Print the minimum required capacity.
  cout << cap;
  return 0;
}
