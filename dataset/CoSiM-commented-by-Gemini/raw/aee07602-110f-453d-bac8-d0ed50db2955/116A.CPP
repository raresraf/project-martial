/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces.
 * @details This program calculates the minimum required capacity of a tram.
 * It simulates the change in the number of passengers at each stop and
 * finds the maximum number of passengers inside the tram at any given time.
 *
 * @see https://codeforces.com/problemset/problem/116/A
 *
 * Algorithm:
 * 1. Read the number of tram stops, `a`.
 * 2. Initialize `jami` (current passengers) to 0. An array `mas` is used to store
 *    the passenger count after each stop.
 * 3. Loop `a` times for each stop.
 * 4. In each stop, read the number of exiting passengers `b` and entering passengers `c`.
 * 5. Update the current passenger count: `jami = jami - b + c`.
 * 6. Store the new passenger count in the `mas` array.
 * 7. After the loop, sort the array of passenger counts.
 * 8. The maximum value in the sorted array (which is the last element) is the
 *    minimum capacity required for the tram.
 *
 * Time Complexity: O(a * log(a)) due to the sorting step. A more optimal O(a) solution
 * could be achieved by keeping track of the maximum passenger count inside the loop
 * instead of storing all counts and sorting later.
 * Space Complexity: O(a) to store the passenger counts at each stop.
 */
#include <algorithm>
#include <iostream>
using namespace std;
// a: number of stops
// b: passengers exiting
// c: passengers entering
// jami: current number of passengers inside the tram
// mas: array to store the passenger count at each stop
int a, b, c, jami = 0, mas[1000000];

int main() {
  // Read the total number of stops.
  cin >> a;

  /**
   * Block Logic: Simulate passenger flow at each stop.
   * This loop updates the number of passengers on the tram after each stop
   * and records the count.
   */
  for (int i = 0; i < a; i++) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> b >> c;
    // Update the total number of passengers.
    jami = jami - b + c;
    // Store the current passenger count.
    mas[i] = jami;
  }

  // Sort the array of passenger counts to find the maximum.
  sort(mas, mas + a);

  // The last element of the sorted array is the maximum passenger count,
  // which represents the minimum required capacity.
  cout << mas[a - 1];
}
