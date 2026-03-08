/**
 * @4a1cbf79-a468-400e-998d-07641d8e22d4/116A.CPP
 * @brief This program calculates the minimum capacity required for a tram
 *        given a sequence of passengers entering and exiting.
 * Algorithm: Iterative simulation of passenger flow, tracking current occupancy and maximum observed occupancy.
 * Time Complexity: O(N) where N is the number of stops, as it involves a single pass through the input.
 * Space Complexity: O(N) for storing the input data in a 2D array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // Declare 'n' to store the number of tram stops.
  // Functional Utility: Read the total number of stops.
  cin >> n;
  // Declare a 2D array 'arr' to store the exit and enter counts for each stop.
  // arr[i][0] = people exiting, arr[i][1] = people entering.
  int arr[n][2];
  // Block Logic: Read the exit and enter counts for each stop into the 2D array.
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 2; j++)
      cin >> arr[i][j];
  // mx: Stores the maximum number of passengers observed in the tram (minimum capacity).
  // sum: Stores the current number of passengers in the tram.
  int mx = INT_MIN, sum = 0; // Initialize sum to 0 and mx to minimum possible integer value.
  // Block Logic: Iterate through each stop to simulate passenger flow and track maximum occupancy.
  // Invariant: 'sum' holds the current number of passengers, 'mx' holds the maximum observed.
  for (int i = 0; i < n; i++) {
    // Action: Update the current number of passengers.
    // Passengers exiting are subtracted, passengers entering are added.
    sum = sum + arr[i][1] - arr[i][0];
    // Action: Update the maximum capacity if current occupancy is higher.
    mx = max(mx, sum);
  }
  // Functional Utility: Output the minimum capacity required for the tram.
  cout << mx;

  return 0; // Indicate successful program termination.
}
