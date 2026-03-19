/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem (Codeforces 116A).
 * @details This program calculates the minimum possible capacity of a tram
 * based on the number of passengers entering and exiting at each stop.
 *
 * Algorithm: Peak Occupancy Tracking
 * The program simulates the passenger flow stop by stop. It maintains a variable
 * for the current number of passengers on the tram ('x') and another for the
 * maximum number of passengers observed so far ('capacity'). At each stop, 'x'
 * is updated by subtracting the exiting passengers and adding the entering ones.
 * If the new 'x' is greater than the current 'capacity', the 'capacity' is updated.
 * After all stops are processed, 'capacity' holds the minimum required capacity.
 *
 * Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <cstdio>
using namespace std;

int main() {
  // n: the number of tram stops.
  int n;
  // a: passengers exiting, b: passengers entering.
  int a, b;
  // x: the current number of passengers on the tram.
  // capacity: the maximum number of passengers on the tram at any point,
  // which is the minimum required capacity.
  int x = 0, capacity = 0;
  
  scanf("%d", &n);
  
  /**
   * Block Logic: Simulate the passenger flow at each stop.
   * Pre-condition: 'n' holds the number of stops to simulate.
   */
  while (n--) {
    scanf("%d%d", &a, &b);
    
    // Update the current number of passengers.
    x = x - a + b;
    
    // If the current occupancy is the highest seen so far, update the capacity.
    if (x > capacity)
      capacity = x;
  }
  
  // Print the calculated minimum required capacity.
  printf("%d", capacity);
}