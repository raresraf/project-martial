/**
 * @file 1366A.CPP
 * @brief Solves a resource allocation puzzle.
 *
 * This program calculates the maximum number of items (shovels) that can be crafted
 * given two types of resources: sticks (`a`) and diamonds (`b`).
 * There are two ways to craft an item:
 * 1. Using 1 stick and 2 diamonds.
 * 2. Using 2 sticks and 1 diamond.
 * The goal is to find the maximum total number of items that can be made.
 *
 * @algorithm
 * Let `a` be the number of sticks and `b` be the number of diamonds.
 * Let `x` be the number of items made with the first recipe (1 stick, 2 diamonds).
 * Let `y` be the number of items made with the second recipe (2 sticks, 1 diamond).
 *
 * We need to maximize `x + y` subject to the constraints:
 * - `x * 1 + y * 2 <= a` (total sticks used)
 * - `x * 2 + y * 1 <= b` (total diamonds used)
 *
 * Adding the two constraints gives:
 * `3x + 3y <= a + b`
 * `3 * (x + y) <= a + b`
 * `x + y <= (a + b) / 3`
 *
 * This shows that the total number of items, `x + y`, is limited by one-third of the
 * total resources.
 *
 * Additionally, you can't make more items than you have of the least numerous resource.
 * If you have `min(a, b)` resources, you can't possibly make more than `min(a, b)` items,
 * as each item requires at least one of each resource type in some combination.
 *
 * Therefore, the maximum number of items is the minimum of these two upper bounds:
 * `max_items = min( (a + b) / 3, min(a, b) )`
 *
 * The code implements this logic.
 *
 * @complexity
 * Time Complexity: O(t), where `t` is the number of test cases. Each test case is solved in O(1).
 * Space Complexity: O(1).
 */
#include <iostream>
#include <algorithm> // For std::swap

using namespace std;

int main() {
  // `t`: number of test cases.
  int t;
  cin >> t;
  
  /**
   * @brief Loop to process each test case.
   */
  while (t--) {
    // `a` and `b`: number of sticks and diamonds.
    long long a, b;
    cin >> a >> b;
    
    // `ans` will store the maximum number of craftable items.
    long long ans = 0;
    
    // Ensure `a` is the smaller of the two quantities for simplicity.
    // This makes `a` equivalent to `min(a, b)`.
    if (a > b)
      swap(a, b);
      
    // First bound: The total number of items cannot exceed one-third of the total resources.
    ans = (a + b) / 3;
    
    // Second bound: The total number of items cannot exceed the quantity of the scarcer resource.
    // Here we take the minimum of the two bounds.
    if (a < ans)
      ans = a;
      
    cout << ans << endl;
  }
  return 0;
}
