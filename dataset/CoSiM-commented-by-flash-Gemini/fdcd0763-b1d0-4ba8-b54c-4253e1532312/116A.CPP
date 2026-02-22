/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit (`a`) and then a certain number enter (`b`). The program efficiently tracks the
 * current number of passengers (`total`) and the maximum number of passengers observed (`mtotal`)
 * without storing all intermediate states in an array.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A: Tram (Optimized Implementation).
 * @details This program calculates the minimum capacity a tram must have to accommodate all passengers
 * at any point during its journey. It simulates 'n' tram stops, where at each stop, a certain number
 * of passengers exit (`a`) and then a certain number enter (`b`). The program efficiently tracks the
 * current number of passengers (`total`) and the maximum number of passengers observed (`mtotal`)
 * without storing all intermediate states in an array.
 * @algorithm Iterative simulation and maximum tracking with $O(1)$ space.
 * Time Complexity: $O(N)$ where N is the number of tram stops.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream for cin/cout, algorithm for max).

using namespace std; // Using the standard namespace to avoid prefixing std::_.
typedef long long ll; // @typedef ll: Alias for long long integer.
typedef pair<int, int> pii; // @typedef pii: Alias for pair of integers.

int main() {
  /**
   * @brief Main function: orchestrates the tram simulation logic.
   */
  int n; // @var n: The total number of tram stops.
  cin >> n; // Reads the total number of tram stops.

  int mtotal = 0; // @var mtotal: Stores the maximum number of passengers encountered in the tram so far.
  int total = 0;  // @var total: Stores the current number of passengers in the tram.

  /**
   * @brief Simulates the tram journey through each stop, updating passenger counts and tracking the maximum.
   * @block_logic Iterates 'n' times, processing passenger changes at each stop.
   * @invariant `total` reflects the current passenger count, and `mtotal` holds the peak passenger count.
   */
  for (int i = 0; i < n; i++) {
    int a, b; // @var a: Number of passengers exiting at the current stop. @var b: Number of passengers entering at the current stop.
    cin >> a >> b; // Reads the number of exiting and entering passengers for the current stop.
    total += b - a; // Updates the current number of passengers in the tram.
    mtotal = max(total, mtotal); // Updates `mtotal` if the current `total` is higher.
  }

  // Functional Utility: Prints the minimum capacity required (maximum number of passengers observed).
  cout << mtotal << endl;
  return 0; // Indicates successful execution.
}
