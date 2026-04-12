/**
 * @file 116A.CPP
 * @brief Simulation of dynamic passenger flow in a multi-station transit system.
 * 
 * Logic: Tracks the instantaneous occupancy of a tram by iteratively processing 
 * exit (j=0) and entry (j=1) events at each station. The peak occupancy defines 
 * the minimum required capacity.
 * 
 * Algorithm: Iterative state-tracking with high-water mark detection.
 * Time Complexity: $O(N)$ where N is the number of stations.
 * Space Complexity: $O(N)$ for the state-history matrix.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * Memory Allocation: Matrix to store input events and subsequent state transitions.
   * arr[i][0]: Passengers exiting at station i.
   * arr[i][1]: Passengers entering at station i.
   */
  int arr[1001][2];
  int n;
  cin >> n;

  // Functional Utility: Reads the raw exit/entry counts for each station.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++)
      cin >> arr[i][j];
  }

  // Running occupancy counter.
  int current_occupancy = 0;

  /**
   * Block Logic: Computes the occupancy state after each event.
   * Logic: Updates the current_occupancy based on exit (subtraction) or entry (addition).
   * Note: The code reuses the input array to store the resulting occupancy states.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      if (j == 0) {
        // Event: Passenger exit.
        current_occupancy -= arr[i][j];
      } else {
        // Event: Passenger entry.
        current_occupancy += arr[i][j];
      }
      // Functional Utility: Re-purposes the array to cache the instantaneous state.
      arr[i][j] = current_occupancy;
    }
  }

  // Functional Utility: Searches the cached states to find the maximum occupancy reached.
  int max_required_capacity = arr[0][0];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      if (max_required_capacity < arr[i][j]) {
        max_required_capacity = arr[i][j];
      }
    }
  }
  
  // Output the minimum capacity sufficient for the entire route.
  cout << max_required_capacity;
  
  return 0;
}
