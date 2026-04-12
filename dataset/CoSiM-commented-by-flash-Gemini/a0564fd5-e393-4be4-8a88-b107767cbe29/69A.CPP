/**
 * @file 69A.CPP
 * @brief Logic for static equilibrium analysis of force vectors in a 3D Euclidean space.
 * 
 * Logic: Calculates the resultant vector of N input force vectors. A state of 
 * equilibrium (YES) is identified if the magnitude of the resultant vector is zero.
 * 
 * Algorithm: Component-wise summation of vector components.
 * Time Complexity: $O(N \times D)$ where N is the number of vectors and D=3 (dimensions).
 * Space Complexity: $O(N \times D)$ for storing the vector sequence.
 */

#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  // Reads the total number of force vectors to be summed.
  int t;
  cin >> t;
  
  // Matrix representing the input vectors (each row is a vector [x, y, z]).
  int arr[t][3];

  /**
   * Block Logic: Populates the force matrix from standard input.
   * Invariant: Each vector has exactly three orthogonal components.
   */
  for (int i = 0; i < t; i++)
    for (int j = 0; j < 3; j++)
      cin >> arr[i][j];

  // Sequence to store the resultant components for each dimension.
  vector<int> res;

  /**
   * Block Logic: Performs dimension-wise reduction (summation) to find the resultant vector.
   * Logic: Interchanges traversal order (column-major) to simplify component summation.
   */
  for (int j = 0; j < 3; j++) {
    int sum = 0;
    for (int i = 0; i < t; i++) {
      sum += arr[i][j];
    }
    res.push_back(sum);
  }

  /**
   * Functional Utility: Evaluates the equilibrium condition (resultant = [0, 0, 0]).
   */
  if (res[0] == 0 && res[1] == 0 && res[2] == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
