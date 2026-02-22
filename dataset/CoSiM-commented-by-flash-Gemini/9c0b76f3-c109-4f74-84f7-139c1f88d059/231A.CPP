/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Map-based Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers a, b, c, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by counting the number of '1' opinions using a map.
 * This is an alternative approach using a map for frequency counting.
 * @algorithm Counting based on map frequency and thresholding.
 * Time Complexity: $O(N \cdot K \log K)$ where N is the number of problems and K is the number of distinct opinion values (max 3 here), due to map operations. More practically, $O(N)$ because K is constant.
 * Space Complexity: $O(K)$ for the map, where K is the number of distinct opinion values (constant).
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A: Team (Map-based Implementation).
 * @details This program processes 'n' problem statements, each with three opinions (represented by integers a, b, c, typically 0 or 1).
 * It determines how many problems will be implemented. A problem is implemented if at least two people agree
 * that it should be implemented, which is robustly checked by counting the number of '1' opinions using a map.
 * This is an alternative approach using a map for frequency counting.
 * @algorithm Counting based on map frequency and thresholding.
 * Time Complexity: $O(N \cdot K \log K)$ where N is the number of problems and K is the number of distinct opinion values (max 3 here), due to map operations. More practically, $O(N)$ because K is constant.
 * Space Complexity: $O(K)$ for the map, where K is the number of distinct opinion values (constant).
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, map).
using namespace std;     // Uses the standard namespace.

/**
 * @brief Global map to store the frequency of opinion values (0 or 1).
 * @details This map is used to count how many of the three opinions for a given problem are '1'.
 *          It is cleared for each problem instance.
 */
map<int, int> m; // @var m: Maps opinion value (key) to its frequency (value).

int main() {
  /**
   * @brief Main function: orchestrates the problem counting logic.
   */
  int n, // @var n: The total number of problem statements to process.
      tot = 0; // @var tot: Counter for the number of problems that will be implemented.

  cin >> n; // Reads the total number of problem statements.

  /**
   * @brief Processes each of the 'n' problem statements.
   * @block_logic Iterates `n` times, processing one problem statement in each iteration.
   * @invariant `n` decreases by 1 in each iteration until it becomes 0. `tot` counts problems meeting criteria.
   */
  while (n--) {
    int a, b, c; // @var a, b, c: Integers representing the opinions of three individuals for a problem (typically 0 or 1).
    cin >> a >> b >> c; // Reads the three opinions for the current problem statement.

    // Functional Utility: Increments the frequency count for each opinion in the map.
    m[a]++; // Increment count for opinion 'a'.
    m[c]++; // Increment count for opinion 'c'.
    m[b]++; // Increment count for opinion 'b'.

    /**
     * @brief Checks if at least two people agreed (opinion '1') to implement the problem.
     * @block_logic Accesses `m[1]` to get the count of '1' opinions.
     * @invariant `tot` is incremented if `m[1]` is 2 or more.
     */
    if (m[1] >= 2) // If the count of '1' opinions is 2 or more.
      tot++;       // Increment the total count of problems to be implemented.
    
    // Functional Utility: Clears the map for the next problem statement.
    m.clear();
  }

  // Functional Utility: Prints the total count of problems that will be implemented.
  cout << tot << "\n";
  return 0; // Indicates successful execution.
}

