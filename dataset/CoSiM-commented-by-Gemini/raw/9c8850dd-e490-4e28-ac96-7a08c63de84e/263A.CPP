/**
 * @file 263A.CPP
 * @brief Calculates the number of moves to bring a '1' to the center of a 5x5 matrix.
 *
 * This program solves a problem often called "Beautiful Matrix". It reads a 5x5
 * matrix that contains exactly one '1' and all other elements as '0's. The goal
 * is to find the minimum number of single-step moves (up, down, left, or right)
 * required to move the '1' to the center of the matrix, which is at position (2, 2)
 * in 0-indexed terms. This is equivalent to calculating the Manhattan distance
 * between the position of the '1' and the center.
 *
 * @algorithm The program first reads the 5x5 matrix and identifies the coordinates
 * (row and column) of the '1'. Then, it calculates the Manhattan distance, which is
 * the sum of the absolute differences of the coordinates, from the '1's position
 * to the center (2, 2).
 *
 * @time_complexity O(1), because the matrix size is fixed at 5x5. The program
 * always performs 25 reads and a constant number of arithmetic operations.
 * @space_complexity O(1), as the matrix size is constant.
 */
#include <bits/stdc++.h> // Includes all standard libraries, common in competitive programming.

// Collection of macros for shorthand looping, common in competitive programming.
#define for0(i, n) for (int i = 0; i < (int)(n); ++i)
#define for1(i, n) for (int i = 1; i <= (int)(n); ++i)
#define forc(i, l, r) for (int i = (int)(l); i <= (int)(r); ++i)
#define forr0(i, n) for (int i = (int)(n)-1; i >= 0; --i)
#define forr1(i, n) for (int i = (int)(n); i >= 1; --i)

// short hand for usual tokens
#define pb push_back
#define fi first
#define se second

// to be used with algorithms that processes a container Eg: find(aint(c),42)
#define aint(x) (x).begin(), (x).end()  // Forward traversal
#define raint(x) (x).rbegin, (x).rend() // reverse traversal

// traversal function to avoid long template definition. Now with C++11 auto
// ainteviates the pain.
#define tr(c, i) for (__typeof__((c)).begin() i = (c).begin(); i != (c).end(); i++)

// find if a given value is present in a container. Container version. Runs in
// log(n) for set and map
#define present(c, x) ((c).find(x) != (c).end())

// find version works for aint containers. This is present in std namespace.
#define cpresent(c, x) (find(aint(c), x) != (c).end())

// Avoiding wrap around of size()-1 where size is a unsigned int.
#define sz(a) int((a).size())

// Macro to speed up C++ I/O operations.
#define FAST_INP ios::sync_with_stdio(false); cin.tie(0);
const unsigned int M = 1000000007;

using namespace std;

// Shorthand type definitions for commonly used types.
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef long long int ll;
typedef vector<int> vint;
typedef vector<vint> vvint;
typedef double ld;

int main() {
  FAST_INP; // Apply fast I/O settings.

  int index_i, index_j; // Variables to store the coordinates of the '1'.
  vector<vector<int>> matrix(5, vector<int>(5, 0));
  
  /**
   * @brief Loop to read the 5x5 matrix and find the position of the '1'.
   * @invariant The outer loop iterates through rows, the inner through columns.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      // When the '1' is found, store its 0-indexed coordinates.
      if (matrix[i][j] == 1) {
        index_i = i;
        index_j = j;
      }
    }
  }

  // The center of the 5x5 matrix is at (2, 2) in 0-indexed coordinates.
  int dx = 2, dy = 2; 
  
  // Calculate the Manhattan distance: |x1 - x2| + |y1 - y2|.
  int ddx = abs(index_i - dx), ddy = abs(index_j - dy);
  
  // Output the minimum number of moves.
  cout << ddx + ddy;
  return 0;
}