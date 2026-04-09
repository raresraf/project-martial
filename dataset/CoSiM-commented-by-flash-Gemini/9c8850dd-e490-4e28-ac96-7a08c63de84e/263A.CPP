/**
 * @file 263A.CPP
 * @brief This program solves the "Beautiful Matrix" problem, which involves a 5x5 matrix
 * with a single '1' and all other elements '0'. The goal is to calculate the minimum
 * number of moves (swaps with adjacent elements) required to move the '1' to the center
 * of the matrix (row 2, column 2, 0-indexed). This is equivalent to finding the Manhattan
 * distance between the '1' and the center.
 *
 * Algorithm: Matrix traversal and Manhattan distance calculation.
 * 1. Read the 5x5 matrix to find the coordinates of the '1'.
 * 2. Calculate the absolute difference between the '1''s row/column and the center row/column (2).
 * 3. Sum these absolute differences to get the Manhattan distance.
 * 4. Print the calculated distance.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns (constant 5x5), effectively O(1).
 * Space Complexity: O(R*C) to store the matrix, effectively O(1).
 */

#include <bits/stdc++.h> // This wont work only for g++ compiler.

#define for0(i, n) for (int i = 0; i < (int)(n); ++i)  // 0 based indexing
#define for1(i, n) for (int i = 1; i <= (int)(n); ++i) // 1 based indexing
#define forc(i, l, r)                                                          
  for (int i = (int)(l); i <= (int)(r);                                        
       ++i) // closed interver from l to r r inclusive
#define forr0(i, n) for (int i = (int)(n)-1; i >= 0; --i) // reverse 0 based.
#define forr1(i, n) for (int i = (int)(n); i >= 1; --i)   // reverse 1 based

// short hand for usual tokens
#define pb push_back
#define fi first
#define se second

// to be used with algorithms that processes a container Eg: find(aint(c),42)
#define aint(x) (x).begin(), (x).end()  // Forward traversal
#define raint(x) (x).rbegin, (x).rend() // reverse traversal

// traversal function to avoid long template definition. Now with C++11 auto
// ainteviates the pain.
#define tr(c, i)                                                               
  for (__typeof__((c)).begin() i = (c).begin(); i != (c).end(); i++)

// find if a given value is present in a container. Container version. Runs in
// log(n) for set and map
#define present(c, x) ((c).find(x) != (c).end())

// find version works for aint containers. This is present in std namespace.
#define cpresent(c, x) (find(aint(c), x) != (c).end())

// Avoiding wrap around of size()-1 where size is a unsigned int.
#define sz(a) int((a).size())

#define FAST_INP                                                               
  ios::sync_with_stdio(false);                                                 
  cin.tie(0);
const unsigned int M = 1000000007;

using namespace std;

// Shorthand for commonly used types
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef long long int ll;
typedef vector<int> vint;
typedef vector<vint> vvint;
typedef double ld;

/**
 * @brief Main function of the program.
 * Reads a 5x5 matrix, finds the '1', and calculates the Manhattan distance to move it to the center (2,2).
 * @pre Input is a 5x5 matrix of integers where exactly one element is '1'.
 * @post Prints the minimum number of moves (Manhattan distance) to bring the '1' to the center.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  FAST_INP;

  // Functional Utility: `index_i` and `index_j` store the row and column of the '1'.
  int index_i, index_j;
  // Functional Utility: Declares a 5x5 matrix initialized with zeros.
  vector<vector<int>> matrix(5, vector<int>(5, 0));
  // Block Logic: Iterates through the 5x5 matrix to read its elements and find the '1'.
  // Invariant: After finding '1', `index_i` and `index_j` will hold its coordinates.
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      // Block Logic: If the current element is '1', store its coordinates.
      if (matrix[i][j] == 1) {
        index_i = i;
        index_j = j;
      }
    }
  }

  // Functional Utility: `dx`, `dy` are the target center coordinates (row 2, col 2).
  // Functional Utility: `ddx`, `ddy` are the absolute differences in row and column from '1' to center.
  int dx = 2, dy = 2;
  int ddx = abs(index_i - dx), ddy = abs(index_j - dy);
  // Functional Utility: Prints the sum of absolute differences, which is the Manhattan distance.
  cout << ddx + ddy;
  return 0;
}
