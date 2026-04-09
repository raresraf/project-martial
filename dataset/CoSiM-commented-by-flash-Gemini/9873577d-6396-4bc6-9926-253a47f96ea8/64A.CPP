/**
 * @file 64A.CPP
 * @brief This program simulates a game played with a sequence of numbers (1, 2, or 3)
 * to determine if the game state is "Infinite" or "Finite", and if finite, calculates a score.
 * The rules for "Infinite" state are based on specific adjacent number combinations.
 * The score calculation is based on transitions between numbers in the sequence.
 *
 * Algorithm: State-based simulation and score accumulation.
 * 1. Read the number of elements (n) in the sequence.
 * 2. Read the sequence of numbers into a vector.
 * 3. During input, check for adjacent '2' and '3' or '3' and '2' to determine an "Infinite" state.
 * 4. If "Infinite" state is detected, print "Infinite".
 * 5. If "Finite", iterate through the sequence to calculate the score based on predefined transition rules.
 * Time Complexity: O(N) where N is the number of elements in the sequence, due to single passes for input and score calculation.
 * Space Complexity: O(N) to store the sequence of numbers.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric> // accumulate(v.begin(), v.end(), 0)
#include <vector>

using namespace std;
#define ll long long

/**
 * @brief Main function of the program.
 * Orchestrates the game simulation, determines its finiteness, and calculates the score if finite.
 * @pre The input sequence consists of integers 1, 2, or 3.
 * @post Prints "Infinite" or "Finite" followed by the calculated score if finite.
 * @return 0 upon successful execution.
 */
int main() {
  int n;
  // Functional Utility: Reads the number of elements in the sequence.
  cin >> n;
  // Functional Utility: Declares a vector to store the sequence of integers.
  vector<int> A(n);
  // Functional Utility: Flag to indicate if the game state becomes "Infinite".
  bool isInfinite = false;
  // Functional Utility: Stores the previously read element for transition checks. Initialized to 1 to correctly handle the first element.
  int prev = 1;
  // Block Logic: Reads the sequence of numbers and simultaneously checks for "Infinite" game conditions.
  // Invariant: At each iteration, `prev` holds the value of the number before `A[i]`.
  for (int i = 0; i < n; i++) {
    cin >> A[i];
    // Block Logic: Checks for the specific adjacent pair patterns that lead to an "Infinite" game state.
    // Pre-condition: `A[i]` is the current element, `prev` is the element before it.
    if ((A[i] == 2 && prev == 3) || (A[i] == 3 && prev == 2))
      isInfinite = true;
    prev = A[i];
  }

  // Block Logic: Determines the output based on whether the game state is "Infinite" or "Finite".
  if (isInfinite)
    cout << "Infinite" << endl;
  else {
    // Functional Utility: Resets `prev` to the first element of the sequence for score calculation.
    prev = A[0];
    // Functional Utility: Accumulates the score for the finite game.
    ll ans = 0;
    // Block Logic: Iterates through the sequence (starting from the second element) to calculate the score.
    // Invariant: `ans` reflects the accumulated score from transitions up to the current element.
    for (int i = 1; i < n; i++) {
      // Block Logic: Applies scoring rules based on the previous element (`prev`).
      // Pre-condition: `prev` is the element `A[i-1]`.
      if (prev == 1) {
        // Block Logic: Specific scoring rules when the previous element was '1'.
        if (A[i] == 2) {
          // Block Logic: Additional rule for '1' followed by '2', checking `A[i-2]`.
          // Pre-condition: `i > 1` ensures `A[i-2]` is a valid index.
          if (i > 1 && A[i - 2] == 3)
            ans += 2;
          else
            ans += 3;
        } else { // A[i] == 3
          ans += 4;
        }
      } else if (prev == 2) { // A[i] can only be 1 or 3
        ans += 3;
      } else { // prev == 3, A[i] can only be 1 or 2
        ans += 4;
      }
      prev = A[i]; // Functional Utility: Updates `prev` for the next iteration.
    }
    cout << "Finite" << endl;
    cout << ans << endl;
  }
  return 0;
}
