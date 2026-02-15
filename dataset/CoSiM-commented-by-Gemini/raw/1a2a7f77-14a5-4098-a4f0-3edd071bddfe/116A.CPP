/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" competitive programming problem.
 * @details The problem is to calculate the minimum required capacity of a tram.
 * The input consists of the number of people exiting and entering at each stop.
 * The minimum capacity must be at least the maximum number of people on the
 * tram at any single point in time.
 *
 * @section algorithm Algorithm
 * This implementation calculates the passenger count after each stop, stores these
 * counts in an array, and then sorts the array to find the maximum value.
 * 1. Read the total number of stops, `num_stops`.
 * 2. A loop runs `num_stops` times, once for each stop.
 * 3. In each iteration:
 *    a. Read the number of exiting (`exiting`) and entering (`entering`) passengers.
 *    b. Update the `current_passengers` count based on these values.
 *    c. Store this `current_passengers` value in an array `capacity_at_stops`.
 * 4. After the loop, sort the `capacity_at_stops` array in non-decreasing order.
 * 5. The last element of the sorted array is the maximum passenger count, which
 *    represents the required minimum tram capacity. This value is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N log N), where N is the number of stops. The complexity is dominated by the sorting step.
 * - Space Complexity: O(N) to store the passenger counts for each stop.
 *   (Note: A more optimal solution with O(N) time and O(1) space exists by tracking the maximum count inside the loop, avoiding the need for storage and sorting).
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int num_stops;
  cin >> num_stops;
  
  int exiting, entering;
  int current_passengers = 0;
  // An array to store the passenger count after each stop.
  // The size is fixed and large enough for the problem constraints.
  int capacity_at_stops[1000000];

  for (int i = 0; i < num_stops; i++) {
    cin >> exiting >> entering;
    // Calculate the number of passengers after the current stop's exchange.
    current_passengers = current_passengers - exiting + entering;
    // Record this count.
    capacity_at_stops[i] = current_passengers;
  }
  
  // Sort the recorded passenger counts to easily find the maximum value.
  sort(capacity_at_stops, capacity_at_stops + num_stops);
  
  // The last element of the sorted array is the maximum passenger count found.
  cout << capacity_at_stops[num_stops - 1];
  
  return 0;
}