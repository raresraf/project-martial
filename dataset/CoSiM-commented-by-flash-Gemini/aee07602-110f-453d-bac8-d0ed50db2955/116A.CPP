/**
 * @file 116A.CPP
 * @brief Solution to a competitive programming problem: Tram capacity.
 *
 * Algorithm: Iterative simulation of passenger count and finding the maximum.
 * Time Complexity: O(N log N), where N is the number of tram stops, due to sorting.
 * Space Complexity: O(N), for storing passenger counts at each stop.
 *
 * This program simulates the passenger count inside a tram over N stops.
 * It calculates the number of passengers after each stop and determines the maximum
 * number of passengers present in the tram at any single moment, which represents
 * the minimum required capacity.
 */

#include <algorithm> // Required for std::sort
#include <iostream>  // Required for std::cin, std::cout
// Functional Utility: Simplifies standard namespace access for competitive programming.
using namespace std;
// Declare variables for total stops (a), passengers exiting (b), passengers entering (c),
// current passenger count (jami), and an array to store passenger counts at each stop (mas).
int a, b, c, jami = 0, mas[1000000];

/**
 * @brief Main function of the program.
 *
 * Reads tram stop data, calculates passenger counts, and determines the minimum required capacity.
 *
 * @return int Exit status of the program.
 */
int main() {
  // Functional Utility: Reads the total number of tram stops.
  cin >> a;
  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes and record counts.
   * Pre-condition: 'a' contains the total number of stops.
   * Invariant: 'jami' accurately reflects the current number of passengers after each stop,
   *            and 'mas' stores this value for later analysis.
   */
  for (int i = 0; i < a; i++) {
    // Functional Utility: Reads the number of passengers exiting (b) and entering (c) at the current stop.
    cin >> b >> c;
    // Functional Utility: Updates the current passenger count.
    jami = jami - b + c;
    // Functional Utility: Stores the current passenger count in the array.
    mas[i] = jami;
  }
  // Functional Utility: Sorts the array of passenger counts to easily find the maximum.
  // Time Complexity for sort: O(N log N)
  sort(mas, mas + a);
  // Functional Utility: Prints the maximum passenger count encountered, which is the last element after sorting.
  cout << mas[a - 1];
}