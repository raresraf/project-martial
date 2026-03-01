/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem using a greedy algorithm.
 *
 * @problem Given a set of 'n' coins, find the minimum number of coins to take
 * such that their sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm The solution uses a greedy strategy: to maximize the sum with the
 * fewest coins, always pick the highest-value coins first.
 * 1. Read all coin values into a `std::vector` and compute their total sum.
 * 2. Sort the vector in descending order.
 * 3. In a loop, continuously take the largest remaining coin (the front element)
 *    and add it to a personal sum (`mySum`).
 * 4. The loop continues until `mySum` is strictly greater than half of the total sum,
 *    which is equivalent to `mySum > total - mySum`.
 * 5. A counter tracks how many coins are taken. This count is the answer.
 *
 * @implementation_note The use of `coins.erase(coins.begin())` inside the loop is
 * inefficient, as it's an O(N) operation. A better approach would be to use an
 * index to iterate through the sorted vector.
 *
 * @time_complexity O(N log N) for the sort, plus O(k*N) for the while loop, where
 * 'k' is the number of coins taken. In the worst case, this approaches O(N^2).
 * For the problem's constraints (N <= 100), this is acceptable.
 * @space_complexity O(N) to store the coins in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  // ans: number of coins taken
  // total: sum of all coins
  // mySum: sum of the coins taken
  int ans = 0, total = 0, mySum = 0;
  vector<int> coins;

  scanf("%d", &n);
  // Read all coins and calculate the total sum.
  for (int i = 0; i < n; i++) {
    int coin;
    scanf("%d", &coin);
    total += coin;
    coins.push_back(coin);
  }
  // Sort the coins in descending order to prepare for the greedy approach.
  sort(coins.begin(), coins.end());
  reverse(coins.begin(), coins.end());

  // Greedily take coins until our sum is more than half the total.
  while (mySum <= float(total) / 2.) {
    // Take the largest available coin.
    mySum += coins.front();
    ans++;
    // Remove the coin from the available pool. This is an O(N) operation.
    coins.erase(coins.begin());
  }
  
  printf("%d\n", ans);

  return 0;
}
