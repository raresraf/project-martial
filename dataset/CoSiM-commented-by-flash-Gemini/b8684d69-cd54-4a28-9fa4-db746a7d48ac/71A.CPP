/**
 * @file 71A.CPP
 * @brief Greedy algorithm for maximizing profit in a resource-constrained assembly problem.
 * 
 * Logic: Calculates the maximum revenue from crafting two types of burgers (Hamburger 
 * and Chicken Burger) given available ingredients: buns (a), beef patties (b), 
 * and chicken patties (c). Each burger requires exactly 2 buns.
 * 
 * Algorithm: Greedy priority selection based on unit price.
 * Time Complexity: $O(T)$ where T is the number of test cases (greedy decisions are $O(1)$).
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>
#define ll long long int
using namespace std;

int main() {
  ll test;
  // Reads total independent test scenarios.
  if (!(cin >> test)) return 0;
  
  while (test--) {
    ll ans = 0, a, b, c, x, y;
    /**
     * Functional Utility: Reads global inventory and price points.
     * a: Buns, b: Beef patties, c: Chicken patties.
     * x: Hamburger price, y: Chicken burger price.
     */
    cin >> a >> b >> c;
    cin >> x >> y;

    /**
     * Block Logic: Greedy execution based on price ranking.
     * Strategy: Prioritize the burger with the higher unit price to maximize revenue.
     */
    if (x >= y) {
      // Case 1: Hamburgers are more profitable or equal.
      ll make_hamb = min(a / 2, b);
      ans += make_hamb * x;
      // Consumption: Reduces bun inventory.
      a -= make_hamb * 2;
      
      // Secondary: Uses remaining buns for lower-profit chicken burgers.
      ll make_chick = min(a / 2, c);
      ans += make_chick * y;
    } else {
      // Case 2: Chicken burgers are more profitable.
      ll make_chick = min(a / 2, c);
      ans += make_chick * y;
      a -= make_chick * 2;
      
      // Secondary: Uses remaining buns for hamburgers.
      ll make_hamb = min(a / 2, b);
      ans += make_hamb * x;
    }
    
    // Output the optimized total revenue for the current inventory.
    cout << ans << endl;
  }

  return 0;
}
/**
 * Note: The original source contained unused global data structures (ar, br, v, M) 
 * and loops that performed manual minimum checks which were replaced by 
 * min() logic for semantic clarity.
 */
