/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * This program determines if a body is in equilibrium given a set of N forces acting upon it.
 * A body is in equilibrium if the vector sum of all forces acting on it is zero.
 * This translates to the sum of all X-components, Y-components, and Z-components being zero.
 * Algorithm: Accumulate sums of force components and check if all sums are zero.
 * Time Complexity: O(N) - where N is the number of forces, as the program iterates through each force once.
 * Space Complexity: O(1) - uses a constant amount of memory for variables regardless of input size.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries

// Macros for common loop patterns and vector operations.
#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)   // Forward loop
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)   // Backward loop
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i) // Iterator loop
#define _all(v) v.begin(), v.end() // Returns iterators to the beginning and end of a container
#define __(v) memset(v, 0, sizeof(v)) // Sets all bytes of a memory block to zero

using namespace std; // Uses the standard namespace

typedef long long LL;       // Alias for long long
typedef unsigned long long ULL; // Alias for unsigned long long

// Template function to add an element to a vector and return the modified vector.
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief Solves the "Young Physicist" problem.
 * Reads N forces, sums their components, and checks for equilibrium.
 */
void solve() {
  int n; // Declares an integer to store the number of forces.
  cin >> n; // Reads the number of forces from standard input.
  
  // Initializes accumulators for the sum of x, y, and z components of all forces.
  int col1 = 0, col2 = 0, col3 = 0; 
  int x, y, z; // Variables to store the x, y, and z components of each force.
  
  /**
   * Block Logic: Iterates through each force, reads its components, and accumulates sums.
   * Invariant: At the start of each iteration, col1, col2, and col3 hold the sums
   *            of the respective components for all forces processed so far.
   */
  _for(i, 0, n) { // Loop N times, once for each force.
    cin >> x >> y >> z; // Reads the three components of the current force.
    col1 += x; // Adds the x-component to the total x-sum.
    col2 += y; // Adds the y-component to the total y-sum.
    col3 += z; // Adds the z-component to the total z-sum.
  }
  
  // Decision Logic: Checks if the body is in equilibrium.
  // The body is in equilibrium if and only if the sum of all x, y, and z components are all zero.
  if (!col1 && !col2 && !col3) // '!' negates, so it checks if col1 == 0 AND col2 == 0 AND col3 == 0.
    cout << "YES"; // If all sums are zero, the body is in equilibrium.
  else
    cout << "NO";  // Otherwise, the body is not in equilibrium.
}

int main() {
  // Functional Utility: Optimizes C++ standard streams for competitive programming.
  // Disables synchronization with C's stdio and unties cin from cout, improving I/O performance.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  
  // Conditional compilation block for local debugging.
#ifdef HIEUNV
  // Functional Utility: Redirects standard input from "input.txt" and standard output to "output.txt"
  // for easier local testing without console interaction.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  
  // Redundant I/O optimization calls, common in competitive programming templates.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  
  solve(); // Calls the function to solve the problem.
  
  return 0; // Indicates successful program execution.
}
