/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 * This program simulates the passenger flow in a tram over several stops
 * to determine the maximum number of passengers on board at any given time.
 * This maximum number represents the minimum capacity the tram must have.
 *
 * Algorithm:
 * 1. Read the number of tram stops, 's'.
 * 2. Initialize 'current' passenger count and 'max' capacity to zero.
 * 3. Loop 's' times, once for each stop:
 *    a. Read the number of passengers exiting ('l') and entering ('e').
 *    b. Update the 'current' passenger count by subtracting those who exit and adding those who enter.
 *    c. Compare the 'current' count with the 'max' capacity seen so far and update 'max' if 'current' is greater.
 * 4. After the loop, print the final 'max' value, which is the minimum required capacity.
 *
 * Time Complexity: O(S), where S is the number of stops, due to the single loop processing each stop.
 * Space Complexity: O(1), as a fixed number of variables are used regardless of the number of stops.
 */
#include <iostream>
using namespace std;
int main() {
  int s = 0;      // Number of stops.
  int e = 0;      // Number of passengers entering at a stop.
  int l = 0;      // Number of passengers leaving at a stop.
  int max = 0;    // Maximum number of passengers on the tram at any point, represents minimum capacity.
  int current = 0; // Current number of passengers on the tram.
  
  // Read the total number of stops.
  cin >> s;

  /**
   * Block Logic: Simulate the passenger count at each stop.
   * Invariant: Before each iteration, 'current' holds the passenger count after the previous stop,
   * and 'max' holds the maximum passenger count observed up to that point.
   */
  while (s--) {
    // Read the number of passengers leaving and entering at the current stop.
    cin >> l >> e;
    // Update the current passenger count.
    current = current + (e - l);
    
    // If the current passenger count exceeds the max recorded so far, update the max.
    if (current >= max)
      max = current;
  }
  
  // Output the minimum required capacity.
  cout << max;
}
