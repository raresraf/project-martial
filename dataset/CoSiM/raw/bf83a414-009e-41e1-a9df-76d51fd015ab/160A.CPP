#include <iostream>

int main() {
  int numCoins = 0;
  int coins[100];
  int twinsSum = 0;
  int yourSum = 0;
  int answer = 0;
  std::cin >> numCoins;
  for (int i = 0; i < numCoins; i++) {
    std::cin >> coins[i];
    twinsSum += coins[i];
  }

  for (int i = 0; i < numCoins - 1; i++) {
    int temp;
    if (coins[i] < coins[i + 1]) {
      temp = coins[i];
      coins[i] = coins[i + 1];
      coins[i + 1] = temp;
      i = -1;
    }
  }

  for (int i = 0; twinsSum >= yourSum; i++) {
    yourSum += coins[i];
    twinsSum -= coins[i];
    answer++;
  }
  std::cout << answer << "\n";
}
