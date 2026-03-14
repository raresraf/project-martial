/**
 * @64b83473-97de-4db5-bfdd-41bd31fe302f/116A.CPP
 * @brief This program solves problem 116A, "Tram".
 * It simulates a tram's journey through 'n' stops. For each stop, it tracks
 * the number of passengers exiting and entering, calculates the current number
 * of passengers, and determines the maximum number of passengers observed
 * in the tram at any single moment.
 *
 * Algorithm: Linear simulation. It iteratively updates the current passenger count
 * after each exit and entry, and continuously tracks the maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(N) for storing the 'a' and 'b' arrays (can be O(1) if processed on the fly).
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to simulate a tram's passenger count and find the maximum.
 * Reads the number of stops, then for each stop, reads passengers exiting and entering,
 * updates current and maximum passenger counts.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n', representing the total number of tram stops.
  cin >> n; // Reads 'n' from standard input.

  // Declares two arrays of size 'n' to store passengers exiting ('a') and entering ('b') at each stop.
  // Note: These arrays are filled sequentially within the loop; their full storage is not strictly necessary.
  int a[n], b[n]; 

  int max = 0;   // Initializes 'max' to 0, which will store the highest passenger count observed.
  int count = 0; // Initializes 'count' to 0, which stores the current number of passengers in the tram.

  /**
   * Block Logic: Simulates the tram's journey through each of the 'n' stops.
   * For each stop, it reads passengers exiting and entering, updates the current
   * passenger count, and then updates the maximum observed if the current count is higher.
   * Invariant: `max` correctly holds the highest passenger count encountered so far.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each stop.
    cin >> a[i]; // Reads passengers exiting at the current stop.
    count -= a[i]; // Inline: Decrements current passenger count by those exiting.
    // cout << count << endl; // Debugging output.
    if (max < count) { // Inline: If the current passenger count (`count`) is greater than the recorded maximum.
      // cout << max << endl; // Debugging output.
      max = count; // Updates 'max' to the new highest count.
    }
    cin >> b[i]; // Reads passengers entering at the current stop.
    count += b[i]; // Inline: Increments current passenger count by those entering.
    // cout << count << endl; // Debugging output.
    if (max < count) { // Inline: If the current passenger count (`count`) is greater than the recorded maximum.
      // cout << max << endl; // Debugging output.
      max = count; // Updates 'max' to the new highest count.
    }
  }

  cout << max << endl; // Prints the maximum number of passengers observed, followed by a newline.
  return 0;            // Indicates successful program termination.
}