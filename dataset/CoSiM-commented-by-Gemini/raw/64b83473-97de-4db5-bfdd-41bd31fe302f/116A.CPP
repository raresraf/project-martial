/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem.
 *
 * @problem A tram makes 'n' stops. At each stop, some passengers exit and some
 * enter. The goal is to find the minimum capacity the tram must have, which is
 * equivalent to the maximum number of passengers on board at any point.
 *
 * @algorithm The code simulates the passenger flow stop by stop. It maintains a
 * running `count` of the current passengers and a `max` value for the peak count.
 * For each stop, it subtracts the exiting passengers, checks for a new max, adds
 * the entering passengers, and checks for a new max again.
 *
 * @implementation_note The check for a new maximum *after* passengers exit is
 * logically redundant, as the passenger count can only decrease at that point and
 * thus cannot set a new peak. The necessary check is the one performed after
 * passengers have entered. The use of arrays `a` and `b` is also unnecessary, as
 * the data could be processed without being stored, optimizing space to O(1).
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(N) to store the inputs in arrays.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  // Note: Using variable-length arrays (VLAs), which are not standard C++.
  int a[n], b[n];
  // `max` will store the required capacity.
  int max = 0;
  // `count` will store the current number of passengers on the tram.
  int count = 0;

  // Loop through each stop.
  for (int i = 0; i < n; i++) {
    // Passengers exit.
    cin >> a[i];
    count -= a[i];
    
    // This check is redundant, as the count has just decreased.
    if (max < count) {
      max = count;
    }

    // Passengers enter.
    cin >> b[i];
    count += b[i];

    // Check for a new peak capacity after passengers have entered.
    if (max < count) {
      max = count;
    }
  }

  cout << max << endl;
  return 0;
}
