/**
 * @file 64A.CPP
 * @brief Solution for a competitive programming problem ("Triangle").
 * This program processes a sequence of shapes (1=circle, 2=triangle, 3=square)
 * and determines if the configuration results in an "Infinite" perimeter or
 * calculates a "Finite" perimeter based on a set of rules for adjacent shapes.
 *
 * @algorithm Single-pass linear scan for both infinite check and finite calculation.
 * @time_complexity O(N), where N is the number of shapes, as the sequence is
 * iterated through a constant number of times.
 * @space_complexity O(N) to store the sequence of shapes.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric> // accumulate(v.begin(), v.end(), 0)
#include <vector>

using namespace std;
#define ll long long

int main() {
  int n;
  cin >> n;
  vector<int> A(n);
  bool isInfinite = false;
  int prev = 1;

  /**
   * @block
   * @brief First pass: Read input and check for the "Infinite" condition.
   * The "Infinite" condition occurs if a triangle (2) and a square (3) are adjacent.
   */
  for (int i = 0; i < n; i++) {
    cin >> A[i];
    if ((A[i] == 2 && prev == 3) || (A[i] == 3 && prev == 2))
      isInfinite = true;
    prev = A[i];
  }

  if (isInfinite)
    cout << "Infinite" << endl;
  else {
    /**
     * @block
     * @brief Second pass: Calculate the finite perimeter based on transition rules.
     * This block is executed only if no "Infinite" condition was found.
     */
    prev = A[0];
    ll ans = 0;
    for (int i = 1; i < n; i++) {
      // Rule set for transitions from a circle (1)
      if (prev == 1) {
        if (A[i] == 2) { // Circle -> Triangle
          // Special case: A square-circle-triangle sequence (3-1-2) has a reduced cost.
          if (i > 1 && A[i - 2] == 3)
            ans += 2;
          else
            ans += 3;
        } else { // Circle -> Square
          ans += 4;
        }
      } else if (prev == 2) { // Rule set for transitions from a triangle (2)
        // Triangle -> Circle
        ans += 3;
      } else { // Rule set for transitions from a square (3)
        // Square -> Circle
        ans += 4;
      }
      prev = A[i];
    }
    cout << "Finite" << endl;
    cout << ans << endl;
  }
  return 0;
}
