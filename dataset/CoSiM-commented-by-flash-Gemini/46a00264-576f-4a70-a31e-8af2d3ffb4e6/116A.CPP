/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (Codeforces 116A).
 *
 * This program simulates a tram journey with multiple stops. At each stop,
 * a certain number of passengers exit and another number enter. The goal
 * is to determine the maximum number of passengers that are on the tram
 * at any single point in time.
 *
 * Algorithm: Simulation. The program iterates through each stop, updates the
 * current number of passengers, and keeps track of the maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to simulate tram passenger movement and find the maximum capacity.
 *
 * Reads the number of stops 'n', and for each stop, reads the number of passengers
 * exiting ('a') and entering ('b'). It calculates the current number of passengers
 * on the tram and keeps track of the maximum value encountered.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare 'n' for the number of stops, 'tram' for current passengers,
  // 'a' for passengers exiting, 'b' for passengers entering, and 'max' for maximum passengers.
  int n, tram = 0, a, b, max = 0;
  // Read the number of stops 'n' from standard input.
  cin >> n;
  /**
   * @brief Block Logic: Simulates tram movement through 'n' stops.
   *
   * Iterates through each tram stop, updates the current number of passengers ('tram'),
   * and compares it with the 'max' observed capacity.
   * Pre-condition: 'n' specifies the total number of stops.
   * Invariant: 'tram' reflects the current passenger count; 'max' stores the peak count.
   */
  for (int i = 0; i < n; i++) {
    // Read passengers exiting ('a') and entering ('b') for the current stop.
    cin >> a >> b;
    // Update the current number of passengers on the tram.
    tram = tram - a + b;
    // Pre-condition: 'tram' holds the passenger count after the current stop.
    // Block Logic: Updates 'max' if 'tram' has a new peak value.
    if (tram > max) {
      max = tram; // Update 'max' if current passenger count is higher.
    }
  }
  // Print the maximum number of passengers observed on the tram.
  cout << max;

  return 0;
}
