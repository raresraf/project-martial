/**
 * @file 1366A.CPP
 * @brief Solves the "Shovels and Swords" problem from Codeforces (Problem 1366A).
 * This program aims to find the maximum number of items (shovels or swords) that can be crafted
 * given a certain amount of iron ore (n) and diamond ore (m). Each shovel requires 2 iron and 1 diamond.
 * Each sword requires 1 iron and 2 diamonds.
 *
 * Algorithm: Greedy Iteration with Resource Normalization. The solution iteratively attempts to craft
 * items by a calculated 'temp' amount, prioritizing a balanced consumption strategy. Resources are
 * normalized such that 'n' always holds the larger quantity of ore, allowing the `min((n / 4) + 1, m)`
 * heuristic to guide resource allocation.
 * Time Complexity: O(log(min(n, m))) or O(sqrt(min(n,m))) due to the nature of the `temp` calculation
 *                  and resource reduction, effectively reducing resources significantly in each step.
 *                  Alternatively, it might be O(max(n,m)) in worst case, but practically much faster.
 * Space Complexity: O(1).
 */

// Problem : A. Shovels and Swords
// Contest : Codeforces - Educational Codeforces Round 89 (Rated for Div. 2)
// URL : https://codeforces.com/contest/1366/problem/A
// Memory Limit : 256.000000 MB
// Time Limit : 1000.000000 milisec

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm, vector).
#define ll long long int // @brief Type definition for long long integers.
#define pb push_back     // @brief Macro for std::vector::push_back.
#define F first          // @brief Macro for std::pair::first.
#define S second         // @brief Macro for std::pair::second.
#define mp make_pair     // @brief Macro for std::make_pair.
#define MOD 1000000007   // @brief Modulo constant, commonly used in competitive programming.
#define vi vector<int>   // @brief Type definition for a vector of integers.
#define vll vector<ll>   // @brief Type definition for a vector of long long integers.
#define pll pair<ll, ll> // @brief Type definition for a pair of long long integers.
#define pii pair<int, int> // @brief Type definition for a pair of integers.
#define all(p) p.begin(), p.end() // @brief Macro for iterating over a container.
#define mid(s, e) (s + (e - s) / 2) // @brief Macro for calculating the midpoint.
#define eb emplace_back // @brief Macro for std::vector::emplace_back.
#define ull unsigned long long // @brief Type definition for unsigned long long integers.
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl; // @brief Debugging macro to print variable name and value.
// @brief Macro to speed up C++ I/O operations (disables sync with C stdio).
#define KAMEHAMEHA ios_base::sync_with_stdio(0);
// @brief Macro to further speed up C++ I/O operations (unties cin/cout).
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
int main() {
  /**
   * @brief Main function to solve the Shovels and Swords problem.
   * Reads multiple test cases. For each case, it determines the maximum number
   * of items that can be crafted given initial iron and diamond ore.
   */
  KAMEHAMEHA // @brief Applies I/O optimization (disables sync with C stdio).
  int t; // @brief Number of test cases.
  cin >> t;
  // Block Logic: Loop through each test case.
  while (t--) {
    ll n, m; // @brief n: iron ore, m: diamond ore.
    cin >> n >> m;
    
    // Block Logic: Normalize resources to ensure 'n' is always the larger quantity.
    // This helps in applying a consistent greedy strategy where the first resource (n)
    // is assumed to be in greater supply or has a specific role.
    if (n < m)
      swap(n, m);
    
    ll total = 0; // @brief Accumulator for the total number of items crafted.

    // Block Logic: Iteratively craft items as long as resources are sufficient.
    // The loop continues as long as there's enough iron for at least one shovel (2 iron)
    // and at least one diamond.
    // Invariant: 'total' accumulates the number of crafted items in each iteration.
    while (n >= 2 && m) {
      // Block Logic: Determine 'temp', the number of items to attempt crafting in this iteration.
      // Heuristic: min((n / 4) + 1, m)
      // `n / 4 + 1` attempts to estimate a quantity based on the iron resource,
      // potentially allowing for some combination of shovels and swords,
      // while `m` limits by diamond availability. This is a heuristic to reduce resources.
      ll temp = min((n / 4) + 1, m);
      
      // Block Logic: Consume resources assuming 'temp' items are crafted.
      // This consumption pattern (2 iron, 1 diamond per 'temp' item) suggests
      // it models crafting 'temp' shovels or a similar resource ratio.
      n -= (temp * 2);
      m -= temp;
      
      // Block Logic: Termination condition if no items can be crafted in this step.
      if (temp == 0)
        break;
      
      total += temp; // Inline: Add crafted items from this step to the total.
      
      // Block Logic: Re-normalize resources after consumption.
      // Maintains 'n' as the larger quantity for the next iteration.
      if (n < m)
        swap(n, m);
    }
    // Block Logic: Output the maximum total number of items that can be crafted.
    cout << total << endl;
  }
  return 0; // @brief Indicates successful program termination.
}
