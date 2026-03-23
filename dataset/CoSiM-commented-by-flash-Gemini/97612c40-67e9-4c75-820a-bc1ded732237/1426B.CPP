/**
 * @file 1426B.CPP
 * @brief This competitive programming solution determines if a specific condition can be met
 *        given a set of 2x2 matrices and an integer 'm'.
 *        The problem essentially checks if it's possible to form a "symmetric" pair within
 *        the matrices when 'm' is even.
 * Algorithm: Iterative check. It first verifies the parity of 'm'. If 'm' is odd, the condition
 *            cannot be met. If 'm' is even, it iterates through all provided 2x2 matrices,
 *            checking if any matrix has its top-right element equal to its bottom-left element.
 * Time Complexity: O(N), where N is the number of 2x2 matrices, as each matrix is processed
 *                  a constant number of times.
 * Space Complexity: O(N), for storing the N 2x2 matrices.
 */
/*
This code is written by Shammi Anand
contact : shammianand101@gmail.com, shammianand.me
*/
#include <bits/stdc++.h> // Includes common standard library headers for competitive programming.
using namespace std;

// Commonly used macros for competitive programming for brevity and convenience.
#define ll long long      ///< Macro for long long integer type.
#define pb push_back      ///< Macro for vector's push_back method.
#define f first           ///< Macro for pair's first element.
#define s second          ///< Macro for pair's second element.
#define vi vector<int>    ///< Macro for vector of integers.
#define pii pair<int, int> ///< Macro for pair of integers.
#define rep(i, a, n) for (int i = a; i < n; i++) ///< Loop macro from 'a' to 'n-1'.
#define F(i, n) for (int i = 0; i < n; i++)      ///< Loop macro from 0 to 'n-1'.
#define all(a) a.begin(), a.end()               ///< Macro for iterator range of a container.
#define INF 1e9 + 7       ///< Macro for a large integer value, often used as infinity.
#define nl "
"           ///< Macro for newline character.
#define w(x)              
  int x;                  
  cin >> x;               
  while (x--)             ///< Macro for handling multiple test cases.

/**
 * @brief Outputs elements of a vector to standard output.
 * @tparam T_vector The type of the vector.
 * @param v The vector to be outputted.
 * @param start The starting index (inclusive) for output. Defaults to 0.
 * @param end The ending index (exclusive) for output. Defaults to vector size.
 */
template <typename T_vector>
void output(const T_vector &v, int start = -1, int end = -1) {
  if (start < 0)
    start = 0;
  if (end < 0)
    end = int(v.size());

  for (int i = start; i < end; i++) {
    cout << v[i] << (i < end - 1 ? ' ' : '
');
  }
}

/**
 * @brief Configures standard I/O for faster execution and redirects input/output
 *        to files if not running in an online judge environment.
 * Functional Utility: Optimizes I/O operations, crucial for competitive programming
 *                     to avoid TLE (Time Limit Exceeded) on large inputs.
 */
void shammi() {
  ios_base::sync_with_stdio(0); // Unties C++ streams from C standard streams.
  cin.tie(0);                   // Unties cin from cout.
  cout.tie(0);                  // Unties cout (though not strictly necessary after cin.tie(0)).
#ifndef ONLINE_JUDGE            // Block Logic: Conditional compilation for local debugging.
  freopen("input.txt", "r", stdin); // Redirects standard input from "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif
}

/**
 * @brief Main function of the program. Reads test cases, processes matrices, and prints results.
 * Functional Utility: Serves as the entry point, orchestrating the problem-solving logic.
 */
int main() {
  shammi(); // Initializes fast I/O and file redirection.
  // Block Logic: Loops through 't' number of test cases. 'w(t)' is a macro that
  //              reads an integer 't' and then creates a while loop that runs 't' times.
  w(t) {
    int n, m; // 'n': number of 2x2 matrices; 'm': an integer used in the condition.
    cin >> n >> m; // Reads the number of matrices and the integer 'm'.
    // Declares an array to store 'n' 2x2 matrices. Each matrix is represented by a 2x2 integer array.
    int matrices[n][2][2];
    // Block Logic: Reads the elements of 'n' 2x2 matrices.
    // Invariant: After this loop, all 'n' matrices are populated with values from input.
    for (int i = 0; i < n; i++) {
      cin >> matrices[i][0][0] >> matrices[i][0][1] >> matrices[i][1][0] >>
          matrices[i][1][1];
    }
    // Block Logic: Checks if 'm' is odd. If so, the condition is always 'NO'.
    if (m % 2) {
      cout << "NO
"; // Prints "NO" followed by a newline.
    } else {          // Block Logic: 'm' is even, so proceed to check matrix properties.
      bool ok = false; // Flag to indicate if a suitable matrix is found.
      // Block Logic: Iterates through each of the 'n' matrices.
      // Invariant: 'ok' will be true if any matrix satisfies the condition.
      for (int i = 0; i < n; i++) {
        // Condition Check: Determines if the top-right element is equal to the bottom-left element
        //                  of the current 2x2 matrix. This implies symmetry along the anti-diagonal.
        if (matrices[i][0][1] == matrices[i][1][0]) {
          ok = true;   // Set flag to true if the condition is met.
          break;       // Exit loop early as one suitable matrix is sufficient.
        }
      }
      // Outputs "YES" if a suitable matrix was found (ok is true), otherwise "NO".
      cout << (ok ? "YES
" : "NO
");
    }
  }
  return 0; // Indicates successful program termination.
}