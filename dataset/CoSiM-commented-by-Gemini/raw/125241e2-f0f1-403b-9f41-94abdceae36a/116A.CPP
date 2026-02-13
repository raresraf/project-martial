/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem (Codeforces 116A).
 * @author competitive-programmer
 *
 * @details This program calculates the minimum required capacity for a tram.
 * The input consists of the number of passengers exiting and entering at each
 * stop. The minimum capacity must be at least the maximum number of passengers
 * inside the tram at any single point in time.
 *
 * Algorithm:
 * 1. Initialize `maxCapacity` (the final answer) and `curCapacity` (current
 *    passenger count) to 0.
 * 2. Read the total number of stops.
 * 3. Loop for each stop.
 * 4. In each iteration, read the number of passengers exiting (`out`) and
 *    entering (`in`).
 * 5. Update the current passenger count: `curCapacity = curCapacity - out + in`.
 * 6. Check if the `curCapacity` is greater than `maxCapacity`. If it is, update
 *    `maxCapacity` with the new peak value.
 * 7. After the loop, `maxCapacity` holds the highest passenger count seen during
 *    the trip, which is the minimum required capacity. Print this value.
 *
 * Time Complexity: O(n), where n is the number of stops.
 * Space Complexity: O(1).
 */
#include <cstdio>
using namespace std;

int main() {
  int numStops;
  scanf("%d\n", &numStops);
  
  // `curCapacity` tracks the number of passengers currently on the tram.
  // `maxCapacity` tracks the peak number of passengers seen so far.
  int curCapacity = 0, maxCapacity = 0, out = 0, in = 0;
  
  /**
   * Block Logic: This loop simulates the tram's journey, processing one stop
   * per iteration.
   */
  while (--numStops >= 0) {
    // Read the number of exiting and entering passengers for the current stop.
    scanf("%d %d\n", &out, &in);
    
    // Update the current passenger count.
    curCapacity += in - out;
    
    // If the current count is a new peak, update maxCapacity.
    if (curCapacity > maxCapacity) {
      maxCapacity = curCapacity;
    }
  }
  
  // Print the minimum capacity required.
  printf("%d\n", maxCapacity);
  return 0;
}
