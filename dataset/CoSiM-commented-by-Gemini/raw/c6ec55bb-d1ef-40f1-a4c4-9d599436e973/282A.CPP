/**
 * @file raw/c6ec55bb-d1ef-40f1-a4c4-9d599436e973/282A.CPP
 * @brief Simulates a series of increment and decrement operations on a variable.
 *
 * This program calculates the final value of a variable `X`, initially 0,
 * after performing a given sequence of operations. The valid operations are
 * "++X", "X++", "--X", and "X--".
 *
 * @algorithm
 * The algorithm initializes a counter variable `X` to zero. It then iterates
 * through a given number of operations. In each iteration, it reads an
 * operation as a string. If the string contains "++", it increments `X`. If it
 * contains "--", it decrements `X`. Finally, it prints the accumulated value of `X`.
 * The distinction between pre- and post-increment/decrement is irrelevant as
 * the operations are standalone statements.
 *
 * @time_complexity O(N), where N is the number of operations, as it involves a
 *                  single loop over the operations.
 * @space_complexity O(1), as storage is constant regardless of the number of operations.
 */
#include <bits/stdc++.h>

using namespace std;

// --- Competitive Programming Boilerplate ---
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
#define nl << "
"

int main() {
  /**
   * @functional
   * Unties C++ streams from C standard streams and disables synchronization
   * to speed up input/output, a common optimization in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int X = 0, N;
  string op;
  cin >> N;

  /**
   * @block
   * Main loop to process `N` operations.
   * Invariant: `X` holds the accumulated value from the operations processed so far.
   */
  for (int i = 0; i < N; i++) {
    cin >> op;
    // Conditional block to modify `X` based on the operation string.
    if (op == "X++")
      X++;
    else if (op == "X--")
      X--;
    else if (op == "++X")
      ++X;
    else if (op == "--X")
      --X;
  }
  // Output the final value of X.
  cout << X << endl;
}
