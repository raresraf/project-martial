/**
 * @file 282A.CPP
 * @brief This program simulates a sequence of increment and decrement operations
 *        on an integer variable 'X'. It processes 'N' operations and outputs
 *        the final value of 'X'.
 * Algorithm: Simulation.
 * Time Complexity: O(N) where N is the number of operations to be processed.
 * Space Complexity: O(1) for storing the variable 'X' and operation string.
 */
#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int X = 0, N; // Initializes X to 0 and declares N for the number of operations.
  string op;    // Variable to store each operation string.
  // Reads the total number of operations to be performed.
  cin >> N;
  /**
   * Block Logic: Processes N operations, updating the value of X based on the operation string.
   * Invariant: In each iteration, one operation is read and applied to X.
   */
  for (int i = 0; i < N; i++) {
    cin >> op; // Reads the current operation.
    if (op == "X++")
      X++; // Post-increment X: X is used, then incremented.
    else if (op == "X--")
      X--; // Post-decrement X: X is used, then decremented.
    else if (op == "++X")
      ++X; // Pre-increment X: X is incremented, then used.
    else if (op == "--X")
      --X; // Pre-decrement X: X is decremented, then used.
  }
  // Prints the final value of X after all operations are processed.
  cout << X << endl;
}
