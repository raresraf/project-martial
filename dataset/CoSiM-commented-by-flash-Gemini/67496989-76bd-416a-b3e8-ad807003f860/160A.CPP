#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int coins[n], total = 0, result = 0;
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
    total += coins[i];
  }

  total /= 2;
  sort(coins, coins + n);

  int sum = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum += coins[i];
    result++;
    if (sum > total)
      break;
  }

  cout << result << endl;

  return 0;
}