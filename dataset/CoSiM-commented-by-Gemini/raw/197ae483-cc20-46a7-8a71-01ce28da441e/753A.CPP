/**
 * @file 753A.CPP
 * @brief Solution to the "Santa Claus and Candies" competitive programming problem.
 * @details The problem asks to represent a given integer 'n' as a sum of the
 * maximum possible number of distinct positive integers.
 *
 * @section algorithm Algorithm
 * A greedy two-phase algorithm is used to solve this problem optimally:
 * 1.  **Initial Greedy Partition:**
 *     a. To maximize the number of summands, the algorithm should use the smallest
 *        possible distinct integers. It starts by creating an initial set of summands
 *        by picking 1, 2, 3, ... sequentially.
 *     b. It continues adding the next integer `p` to a vector and subtracting `p` from `n`
 *        until `n` is less than the next integer `p+1`.
 *     c. This results in an initial set `{1, 2, ..., k}` and a small remainder `rem` (now stored in `n`).
 *
 * 2.  **Remainder Distribution:**
 *     a. The remainder `rem` is then distributed among the integers already in the set.
 *     b. To maintain the distinctness of the integers, the remainder is added one by one
 *        to the largest numbers in the set. The algorithm iterates backwards from the largest
 *        element, incrementing it by one and decrementing the remainder.
 *     c. This works because incrementing the largest numbers first ensures they stay distinct
 *        from their smaller neighbors. For example, if we have {5, 6} and a remainder of 1,
 *        incrementing 6 to 7 results in {5, 7}, which is still distinct.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(sqrt(N)). The first loop runs approximately sqrt(N) times because the sum of the first k integers is k*(k+1)/2. The second loop runs at most k times, which is also on the order of O(sqrt(N)).
 * - Space Complexity: O(sqrt(N)) to store the resulting list of integers.
 */
#include <iostream>
#include <vector>

int main() {
  long n;
  std::cin >> n;
  std::vector<long> a;

  // Phase 1: Greedily build an initial set of distinct summands (1, 2, 3, ...).
  // The loop continues as long as the next integer 'p' is not greater than the remaining 'n'.
  for (long p = 1; p <= n; p++) {
    a.push_back(p);
    n -= p;
  }
  
  // At this point, 'n' holds the remainder from the initial partition.
  
  // Phase 2: Distribute the remainder 'n' among the largest elements in the vector 'a'
  // to maintain the distinctness property.
  for (long p = a.size() - 1; p >= 0 && n > 0; p--) {
    // Add 1 to the current largest available element.
    ++a[p];
    // Decrement the remainder.
    --n;
  }

  // Print the result: the number of summands and the summands themselves.
  std::cout << a.size() << std::endl;
  for (long p = 0; p < a.size(); p++) {
    std::cout << a[p] << " ";
  }
  std::cout << std::endl;

  return 0;
}
