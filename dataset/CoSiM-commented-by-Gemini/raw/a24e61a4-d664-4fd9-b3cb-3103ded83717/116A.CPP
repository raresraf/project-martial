/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem, which calculates the minimum required capacity of a tram.
 *
 * This program simulates the passenger flow in a tram over a number of stops
 * to determine the maximum number of passengers inside the tram at any given time.
 * This maximum number represents the minimum capacity the tram must have.
 *
 * @algorithm Simulation
 * @time_complexity O(N), where N is the number of tram stops. The program iterates through the stops twice (once for input, once for simulation).
 * @space_complexity O(N) to store the passenger exit and entry data for each stop.
 */
#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int l;
  cin >> l;

  int A[1000], B[1000];
  /**
   * @brief Reads the passenger data for each stop.
   * A[i] = passengers exiting at stop i.
   * B[i] = passengers entering at stop i.
   */
  for (int i = 0; i < l; i++) {
    cin >> A[i] >> B[i];
  }

  int a = 0; // 'a' tracks the current number of passengers in the tram.
  int b = 0; // 'b' tracks the maximum number of passengers seen so far.
  
  /**
   * @brief Simulates the passenger flow through all the stops.
   *
   * Pre-condition: Passenger data for all stops is loaded into arrays A and B.
   * Invariant: At the end of each iteration `i`, `a` holds the number of passengers
   * after the events at stop `i`, and `b` holds the maximum value of `a` seen up to stop `i`.
   */
  for (int i = 0; i < l; i++) {
    // Update the current passenger count: subtract exiting, add entering.
    a = (a - A[i]) + B[i];
    // Update the maximum capacity required.
    b = max(a, b);
  }

  // Output the minimum required capacity.
  cout << b << endl;
}