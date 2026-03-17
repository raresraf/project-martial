/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 116A "Tram".
 *
 * @details
 * This program calculates the minimum possible capacity of a tram. It is given the
 * number of tram stops and, for each stop, the number of passengers exiting and
 * entering the tram. The goal is to find the maximum number of passengers inside
 * the tram at any single point in time.
 *
 * @algorithm
 * The algorithm simulates the passenger flow stop by stop.
 * 1. It reads all the exit and entry data into two separate arrays first.
 * 2. It initializes a `currentPassengers` count and a `max` capacity tracker to zero.
 * 3. It then iterates through each stop, updating the `currentPassengers` by
 *    subtracting those who exit and adding those who enter.
 * 4. After each update, it compares the `currentPassengers` with the `max` value
 *    seen so far and updates `max` if the current count is higher.
 * 5. The final `max` value is the minimum required capacity.
 *
 * @note The implementation uses arrays to store all input before processing. A more
 * space-efficient approach with O(1) space would be to process the data for each
 * stop as it's read, without storing the entire history.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stops, due to the linear scan
 *   through the stops.
 * - Space Complexity: O(N), for storing the exit and entry counts in arrays.
 */

#include <algorithm>
#include <iostream>

using namespace std;

#define ll long long;

// Global variables to store the number of stops and passenger counts.
int n, a[1000], b[1000];

/**
 * @brief Executes the core simulation logic.
 */
void solve() {
  int max = 0, currentPassengers = 0;

  /**
   * @brief Simulates the passenger count at each stop.
   */
  for (int i = 0; i < n; i++) {
    // Update the number of passengers after exits and entries at the current stop.
    currentPassengers += b[i] - a[i];
    
    // Track the maximum number of passengers seen so far.
    if (max < currentPassengers)
      max = currentPassengers;
  }
  
  // The maximum number of passengers at any time is the required capacity.
  cout << max;
}

/**
 * @brief Main function, handles input and calls the solver.
 */
int main() {
  // Read the total number of stops.
  cin >> n;
  
  /**
   * @brief Reads the exit (a[i]) and entry (b[i]) counts for all stops.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  
  solve();
}
