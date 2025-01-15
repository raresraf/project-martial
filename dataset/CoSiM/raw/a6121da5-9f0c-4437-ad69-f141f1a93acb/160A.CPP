#include <algorithm>
#include <iostream>
#include <vector>
/* The basic idea is to sort the coins in ascending order and start taking coins
from the opposite end until we reach a point where the amount we have is
strictly greater than the amount remaining. */
int main() {
  int number_of_coins, total_sum = 0, minimum_number_of_coins = 0, temp_sum = 0;
  std::vector<int> all_coins;
  std::cin.sync_with_stdio(false); // For faster I/O
  std::cin >> number_of_coins;
  for (int i = 0; i < number_of_coins; i++) {
    int coin_value;
    std::cin >> coin_value;
    total_sum += coin_value; // Keeping total sum for later computation
    all_coins.push_back(coin_value);
  }
  sort(all_coins.begin(), all_coins.end()); // Sorting the vector
  for (int i = all_coins.size() - 1; i >= 0; i--) {
    minimum_number_of_coins += 1; // Keep adding coins
    temp_sum += all_coins[i];     // Compute sum of all coins taken till now
    if (temp_sum >
        total_sum - temp_sum) { // Whenever sum of coins taken more than the sum
                                // of remaining coins, break
      break;
    }
  }
  std::cout << minimum_number_of_coins;
}
