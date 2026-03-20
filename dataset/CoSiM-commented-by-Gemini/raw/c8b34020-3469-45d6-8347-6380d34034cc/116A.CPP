/**
 * @file 116A.CPP
 * @brief Solution for Codeforces problem 116A, "Tram".
 * @details This program calculates the minimum required capacity of a tram by tracking
 * the number of passengers at each stop to find the peak occupancy.
 *
 * @section Algorithm
 * The problem is solved by simulating the passenger count inside the tram as it
 * moves between stops. The minimum required capacity is the maximum number of
 * passengers present in the tram at any single point in time.
 *
 * The algorithm is as follows:
 * 1. Read the total number of tram stops.
 * 2. Store the number of passengers exiting and entering at each stop.
 * 3. Initialize a `current_passengers` counter and a `max_capacity` tracker to zero.
 * 4. Iterate through each stop:
 *    a. Update `current_passengers` by subtracting the exiting passengers and adding
 *       the entering ones.
 *    b. After updating, compare `current_passengers` with `max_capacity` and update
 *       `max_capacity` if the current count is greater.
 * 5. The final value of `max_capacity` is the minimum capacity required.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of stops, as the program
 *   iterates through the stops twice (once for reading, once for calculating).
 * - Space Complexity: O(n), as a vector is used to store the data for all stops.
 *   This could be optimized to O(1) by calculating the capacity while reading the input.
 */
#include <bits/stdc++.h>

using namespace std;

int max_capacity(vector<pair<int, int>> &tram_status, int n_trams);

int main() {
  int n_trams = 0, element1 = 0, element2 = 0;
  vector<pair<int, int>> tram_status = {};
  pair<int, int> t;

  cin >> n_trams;

  /**
   * @brief Reads the data for each stop, consisting of the number of people
   * exiting and entering the tram.
   */
  while (n_trams--) {
    cin >> element1;
    t.first = element1; // Exiting passengers
    cin >> element2;
    t.second = element2; // Entering passengers
    tram_status.push_back(t);
  }

  // Call the function to calculate and then print the minimum required capacity.
  cout << max_capacity(tram_status, tram_status.size());

  return 0;
}

/**
 * @brief Calculates the minimum tram capacity required.
 * @param tram_status A vector of pairs, where each pair represents a stop.
 *        The first element is the number of exiting passengers, and the second
 *        is the number of entering passengers.
 * @param n_trams The total number of stops.
 * @return An integer representing the minimum required capacity.
 */
int max_capacity(vector<pair<int, int>> &tram_status, int n_trams) {
  // MAX_CAPACITY tracks the peak passenger count found so far.
  // diference tracks the current number of passengers in the tram.
  int MAX_CAPACITY = 0, diference = 0;

  /**
   * @brief This loop simulates the passenger flow at each stop to find the peak.
   * Pre-condition: `diference` is 0, representing an empty tram at the first stop before anyone boards.
   * Invariant: At the end of iteration `k`, `diference` holds the passenger count
   * after the k-th stop, and `MAX_CAPACITY` is the maximum count observed up to that point.
   */
  for (int k = 0; k < n_trams; ++k) {
    // Update the current number of passengers.
    diference = diference - tram_status[k].first + tram_status[k].second;
    // Check if the current passenger count is a new maximum.
    MAX_CAPACITY = max(MAX_CAPACITY, diference);
  }

  return MAX_CAPACITY;
}