/**
 * @file 116A.CPP
 * @brief Logic for calculating the minimum required capacity of a public transit vehicle.
 * 
 * Algorithm: One-pass state tracking of passenger inflow and outflow.
 * Logic: Tracks the instantaneous passenger count at each station and identifies 
 * the peak occupancy (high-water mark).
 * 
 * Time Complexity: $O(N)$ where N is the number of stations.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <iostream>

using namespace std;

int main() {
  int n, a, b;
  // State: Current number of entities in the vehicle.
  int passengers = 0;
  // High-water mark: Maximum occupancy observed during the simulation.
  int capacity = 0;
  
  // Reads the total number of stations in the route.
  cin >> n;

  /**
   * Block Logic: Iteratively updates the vehicle state at each station.
   * Invariant: Total passengers can never be negative (physical constraint).
   */
  for (int i = 0; i < n; i++) {
    // a: Count of exiting entities, b: Count of entering entities.
    cin >> a >> b;
    // Functional Utility: Resolves the net change in occupancy at the station.
    passengers -= a;
    passengers += b;
    
    // Logic: Updates the global peak if current occupancy exceeds previous maximum.
    if (passengers > capacity)
      capacity = passengers;
  }

  // Output the minimum capacity sufficient to accommodate all passengers.
  cout << capacity;

  return 0;
}
