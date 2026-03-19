// Problem: Find the Manhattan distance of the number '1' from the center of a 5x5 matrix.
// This is a competitive programming problem involving matrix traversal and basic geometry.
// The solution finds the coordinates of the '1' and then calculates the distance.
// The Manhattan distance between two points (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.
// Time Complexity: O(1), as the matrix size is fixed at 5x5.
// Space Complexity: O(1).
#include <cstdlib>
#include <iostream>
using namespace std;
int main() {
  int i, j;
  // Iterate through a 5x5 grid to find the location of the value '1'.
  for (int x = 0; x < 5; x++) {
    for (int y = 0; y < 5; y++) {
      int t;
      cin >> t;
      if (t == 1) {
        // Store the 1-based coordinates of '1'.
        i = x + 1;
        j = y + 1;
        // Since there is only one '1', we can stop searching.
        break;
      }
    }
  }

  // cout << i << j << endl;

  // Calculate the Manhattan distance from the center of the matrix (3, 3)
  // to the location of '1'.
  cout << abs(3 - i) + abs(3 - j) << endl;
}
