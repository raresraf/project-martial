/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces contest 116.
 *
 * @details
 * The problem asks to calculate the minimum possible capacity of a tram. The tram makes `n` stops.
 * At each stop, a certain number of passengers exit and a certain number enter. The tram is empty
 * before the first stop and after the last stop. The goal is to find the maximum number of passengers
 * inside the tram at any point in time, as this determines the minimum capacity required.
 *
 * The program reads the number of stops `n`. It then iterates `n` times, once for each stop.
 * In each iteration, it reads the number of passengers exiting (`b`) and entering (`a`). It maintains
 * a running count of the passengers currently on the tram (`min`). After each stop, it updates this
 * count and compares it with the maximum number of passengers seen so far (`ans`), updating the
 * maximum if the current number is greater.
 *
 * Algorithm:
 * 1. Read the number of tram stops, `n`.
 * 2. Initialize `current_passengers` to 0.
 * 3. Initialize `max_capacity` to 0.
 * 4. Loop `n` times:
 *    a. Read the number of exiting passengers, `exit_count`.
 *    b. Read the number of entering passengers, `enter_count`.
 *    c. Update `current_passengers = current_passengers - exit_count + enter_count`.
 *    d. Update `max_capacity = max(max_capacity, current_passengers)`.
 * 5. Print `max_capacity`.
 *
 * Time Complexity: O(n), where `n` is the number of stops, because we process each stop's data once.
 * Space Complexity: O(1), as we only use a few variables to store the state.
 */
#include "bits/stdc++.h"
// PRAY :DOLPHINGARLIC: orz orz orz
#define GS                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(nullptr);                                                            \
  cout.tie(0)
#define Int long long
#define all(x) x.begin(), x.end()
#define el '\n'
#define ints unsigned
#define Ints unsigned long long
#define elif else if
#define test                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

using namespace std;

int main() {
  // GS is a macro for optimizing C++ I/O operations.
  GS;
  int n;
  // Read the number of tram stops.
  cin >> n;
  int ans = 0; // `ans` will store the maximum number of passengers, i.e., the minimum required capacity.
  int as = n * 2; // This variable seems to be incorrectly calculated and used. The loop condition should be `i < n`.
  int min{}; // `min` tracks the current number of passengers inside the tram.
  
  // Block Logic: Iterate through each stop to calculate passenger flow.
  // The loop should run 'n' times, once for each stop. The original code has a bug in the loop condition.
  for (int i = 0; i < n; i++) {
    int b{}; // Passengers exiting at the current stop.
    cin >> b;
    int a{}; // Passengers entering at the current stop.
    cin >> a;
    
    // Functional Utility: Update the passenger count.
    min -= b; // Passengers exit.
    min += a; // Passengers enter.
    
    // Pre-condition: Check if the current number of passengers exceeds the maximum recorded so far.
    if (ans < min) {
      // If so, update the maximum capacity required.
      ans = min;
    }
  }
  // Output the minimum capacity required for the tram.
  cout << ans << endl;
  return 0;
}
/*
1:25 OM
9/4/2020
*/
