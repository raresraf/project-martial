/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins for a majority sum.
 *
 * The problem is to determine the minimum number of coins one needs to take from a
 * collection so that the sum of the taken coins is strictly greater than the sum
 * of the remaining coins.
 *
 * @algorithm Greedy Approach
 * @details The optimal strategy is to always pick the coins with the largest values
 * first. We sort the coins in descending order and take them one by one until their
 * sum is strictly greater than half of the total sum.
 *
 * @complexity
 * Time: O(N log N), dominated by the sorting of the coins.
 * Space: O(N) to store the coin values in a vector.
 */
#include <cstdio>
#include <vector>
#include <numeric>   // For std::accumulate (though not used, sum is manual)
#include <algorithm> // For std::sort

using namespace std;

/**
 * @brief Custom comparison function to sort integers in descending order.
 * @return True if 'a' is greater than 'b', false otherwise.
 */
bool cmp(const int &a, const int &b) { return a > b; }

/**
 * @brief Main function to handle input and execute the greedy algorithm.
 */
int main() {
  int n;       // The total number of coins.
  int ai;      // A temporary variable to read each coin's value.
  int sum = 0; // The total value of all coins.
  int ansS = 0; // The sum of coins taken by our twin.
  int ans = 0;  // The count of coins taken by our twin.
  vector<int> coin;

  scanf("%d", &n);

  // Pre-condition: 'coin' vector is empty, 'sum' is 0.
  // Block Logic: Read each coin's value, add it to the vector, and accumulate the total sum.
  for (int i = 0; i < n; i++) {
    scanf("%d", &ai);
    coin.push_back(ai);
    sum += ai;
  }
  
  // Sort the coins in descending order to apply the greedy strategy.
  sort(coin.begin(), coin.end(), cmp);

  // Pre-condition: 'ansS' (our sum) and 'ans' (our count) are 0.
  // Block Logic: Greedily take the largest coins until our sum is strictly
  // greater than the remaining sum. The condition `ansS <= sum / 2` is an
  // efficient way to check this. We stop as soon as our sum exceeds half.
  for (int i = 0; i < n && ansS <= sum / 2; i++) {
    ansS += coin[i];
    ans++;
  }
  
  // Print the minimum number of coins required.
  printf("%d", ans);
  
  return 0;
}
