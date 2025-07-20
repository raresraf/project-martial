#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int coins[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
    sum += coins[i];
  }
  sort(coins, coins + n, greater<int>());
  int min, tot = 0;
  for (min = 0; tot <= sum; min++) {
    tot += coins[min];
    sum -= coins[min];
  }
  cout << min;
}