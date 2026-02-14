/**
 * @file 753A.CPP
 * @brief Solution for the "Santa Claus and Candies" problem.
 * @details The goal is to find the maximum number of distinct positive integers
 * that sum up to a given integer `n`. The program must output the count of these
 * integers and the integers themselves.
 *
 * @section algorithm Algorithm
 * The problem is solved using a two-phase greedy strategy.
 *
 * Phase 1: Initial Distribution
 * 1. Start with an empty set of numbers.
 * 2. Greedily take the smallest possible distinct positive integers (1, 2, 3, ...)
 *    and add them to the set, subtracting their value from the total `n`.
 * 3. This continues until the remaining `n` is less than the next integer in the
 *    sequence. At this point, we have the maximum possible number of summands,
 *    but their sum is not yet equal to the original `n`.
 *
 * Phase 2: Distributing the Remainder
 * 1. The value of `n` left over from Phase 1 is the remainder.
 * 2. To maintain the same number of summands while reaching the target sum, we must
 *    add this remainder to the numbers already in our set.
 * 3. To keep the numbers distinct, we distribute the remainder by adding 1 to the
 *    largest numbers in the set, working backwards. This ensures that the numbers
 *    remain unique. For example, if the set is {1, 2, 3} and the remainder is 2,
 *    we add 1 to 3 (set becomes {1, 2, 4}) and 1 to 2 (set becomes {1, 3, 4}).
 *
 * @section complexity Complexity Analysis
 * Let k be the number of summands. k is approximately sqrt(2n).
 * - Time Complexity: O(sqrt(n)), as both loops (for initial distribution and
 *   remainder distribution) run a number of times proportional to k.
 * - Space Complexity: O(sqrt(n)) to store the `k` summands in the vector.
 */
#include <iostream>
#include <vector>

/**
 * @brief Main function to solve the candy distribution problem.
 * @return 0 on successful execution.
 */
int main() {
  long n;
  std::cin >> n;
  std::vector<long> a;

  // Phase 1: Greedily build a sequence of 1, 2, 3, ...
  // This loop determines the maximum number of distinct summands (k).
  for (long p = 1; p <= n; p++) {
    a.push_back(p);
    n -= p; // Decrease n by the summand.
  }

  // Phase 2: Distribute the remainder 'n' among the largest elements.
  // This is done to make the sum correct while keeping the elements distinct.
  for (long p = a.size() - 1; p >= 0 && n > 0; p--) {
    a[p]++; // Add 1 to the largest available number.
    n--;    // Decrement the remainder.
  }

  // Output the number of summands.
  std::cout << a.size() << std::endl;
  // Output the final set of distinct summands.
  for (long p = 0; p < a.size(); p++) {
    std::cout << a[p] << " ";
  }
  std::cout << std::endl;

  return 0;
}
