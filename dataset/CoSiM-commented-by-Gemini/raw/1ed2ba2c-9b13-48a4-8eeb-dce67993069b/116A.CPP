/**
 * @file 116A.CPP
 * @brief Solution for Codeforces problem 116A, "Tram".
 *
 * @author Isanchez_Aguilar
 *
 * @description
 * This program calculates the minimum required capacity of a tram based on the
 * number of passengers entering and exiting at each of its `n` stops. The tram
 * starts empty. The goal is to find the maximum number of people inside the
 * tram at any given time.
 *
 * @algorithm
 * 1. Initialize `ans` (maximum capacity needed) and `curPeople` (current
 *    occupancy) to 0.
 * 2. Read the number of stops, `n`.
 * 3. Loop `n` times to simulate each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current occupancy: `curPeople = curPeople - a + b`.
 *    c. Update the maximum capacity required: `ans = max(ans, curPeople)`.
 * 4. After the loop, `ans` holds the peak occupancy, which is the minimum
 *    required capacity. Print `ans`.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  int n;
  cin >> n;

  // `ans` stores the maximum number of people ever on the tram (the required capacity).
  int ans = 0;
  // `curPeople` tracks the current number of people on the tram after each stop.
  int curPeople = 0;

  // Block Logic: Simulate the tram's journey through 'n' stops.
  for (int i = 0; i < n; ++i) {
    int a, b; // a: exiting, b: entering
    cin >> a >> b;

    // Update the current number of people on the tram.
    curPeople -= a;
    curPeople += b;

    // Keep track of the peak number of people encountered so far.
    ans = max(ans, curPeople);
  }

  cout << ans << "\n";

  return 0;
}
