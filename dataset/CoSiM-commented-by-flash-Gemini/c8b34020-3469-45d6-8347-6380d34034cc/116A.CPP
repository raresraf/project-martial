/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem (Codeforces 116A).
 *   It determines the minimum capacity a tram must have to accommodate all passengers
 *   at any point in time during its journey. Passengers enter and exit at `n` stops.
 *   Problem description: https://codeforces.com/problemset/problem/116/A
 * * Algorithm: Cumulative sum (prefix sum) tracking.
 *   The program simulates the tram's journey, maintaining a running total of passengers
 *   currently inside the tram. At each stop, it updates this total and tracks the maximum
 *   number of passengers observed simultaneously, which represents the minimum required capacity.
 * Time Complexity: O(N) where N is the number of tram stops, as it involves reading N inputs
 *   and then iterating through them once.
 * Space Complexity: O(N) for storing the `tram_status` vector of pairs.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.

using namespace std;

/**
 * @brief Calculates the maximum capacity required for a tram.
 * Functional Utility: Simulates the tram's journey, tracking the number of passengers
 *                     inside at each stop and determining the peak occupancy.
 *
 * @param tram_status A vector of pairs, where each pair represents (passengers_exit, passengers_enter)
 *                    at a given stop.
 * @param n_trams The number of stops in the tram journey.
 * @return The maximum number of passengers observed in the tram, which is the minimum required capacity.
 */
int max_capacity(vector<pair<int, int>> &tram_status, int n_trams);

/**
 * @brief Main function to handle input and orchestrate the capacity calculation.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     stores this data, and then calls `max_capacity` to get and print the result.
 */
int main() {
  int n_trams = 0, element1 = 0, element2 = 0; // `n_trams` for total stops, `element1/2` for temporary input.
  vector<pair<int, int>> tram_status = {}; // Stores the (exit, enter) passenger counts for each stop.
  pair<int, int> t; // Temporary pair to read input for each stop.

  // Block Logic: Reads the total number of tram stops.
  cin >> n_trams;

  /**
   * Block Logic: Reads passenger exit and entry data for each tram stop.
   * Pre-condition: `n_trams` is a non-negative integer.
   * Invariant: In each iteration, `element1` (exit) and `element2` (enter) are read for a stop,
   *            and stored as a pair in `tram_status`.
   * Inline: `n_trams--` serves as the loop condition and decrements the counter.
   */
  while (n_trams--) {
    cin >> element1;
    t.first = element1; // Stores passengers exiting.
    cin >> element2;
    t.second = element2; // Stores passengers entering.
    tram_status.push_back(t); // Adds the pair to the vector.
  }

  // Block Logic: Calls `max_capacity` to calculate the result and prints it to standard output.
  cout << max_capacity(tram_status, tram_status.size());

  return 0;
}

int max_capacity(vector<pair<int, int>> &tram_status, int n_trams) {
  int MAX_CAPACITY = 0; // Stores the maximum number of passengers observed in the tram.
  int diference = 0;    // Tracks the current number of passengers inside the tram.

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger flow and track occupancy.
   * Pre-condition: `tram_status` contains valid (exit, enter) pairs for all stops.
   * Invariant: `diference` is updated with current passengers, and `MAX_CAPACITY` reflects the peak.
   */
  for (int k = 0; k < n_trams; ++k) {
    // Functional Utility: Updates the current number of passengers (`diference`).
    // Passengers exiting (`first`) are subtracted, and passengers entering (`second`) are added.
    diference = diference - tram_status[k].first + tram_status[k].second;
    // Functional Utility: Updates `MAX_CAPACITY` to be the maximum of its current value and the current `diference`.
    // This effectively finds the peak occupancy, which is the minimum required capacity.
    MAX_CAPACITY = max(MAX_CAPACITY, diference);
  }

  return MAX_CAPACITY;
}