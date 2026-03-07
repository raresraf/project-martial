/*
 * tram.cpp
 * http://codeforces.com/problemset/problem/116/A
 *  Created on: Oct 26, 2017
 *      Author: athena
 * 31757462 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 18:52:15 	2017-10-26 18:52:16
 */

/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" competitive programming problem (Codeforces 116A).
 * The program simulates a tram journey with multiple stops. At each stop,
 * some passengers exit and others enter. The goal is to determine the minimum
 * capacity the tram must have to accommodate all passengers at any point
 * during its journey.
 * Algorithm: Iterative tracking of current passenger count and maximum seen count.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <iostream> // For input/output operations.
using namespace std;

int main() {
  int n; // Declares 'n' to store the total number of tram stops.
  // Reads the total number of tram stops.
  cin >> n;

  int a, b; // Declares 'a' for passengers exiting and 'b' for passengers entering at a stop.

  int counter = 0; // 'counter' keeps track of the current number of passengers in the tram.
  int max = 0;     // 'max' stores the maximum number of passengers observed at any point.

  /**
   * Block Logic: Iterates through each tram stop to simulate passenger changes and track maximum capacity.
   * Invariant: At the start of each iteration, 'i' is the current stop index,
   * 'counter' is the passenger count before this stop, and 'max' is the maximum
   * passenger count observed up to the previous stop.
   * Pre-condition: 'n' is a positive integer.
   * Post-condition: 'max' holds the overall maximum passenger count during the journey.
   */
  for (int i = 0; i < n; i++) {
    // Reads the number of passengers exiting ('a') and entering ('b') for the current stop.
    cin >> a >> b;
    counter -= a; // Passengers exit the tram.
    counter += b; // Passengers enter the tram.
    
    // Conditional Logic: Updates 'max' if the current total passenger count is higher.
    if (counter > max) {
      max = counter;
    }
  }

  // Prints the minimum required tram capacity.
  cout << max << endl;

  return 0; // Indicates successful program execution.
}
