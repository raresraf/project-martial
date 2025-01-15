#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  int n, ans = 0, total = 0, mySum = 0;
  vector<int> coins;
  scanf("%d", &n);
  for (int i = 0; i < n; i++) {
    int coin;
    scanf("%d", &coin);
    total += coin;
    coins.push_back(coin);
  }
  sort(coins.begin(), coins.end());
  reverse(coins.begin(), coins.end());
  while (mySum <= float(total) / 2.) {
    mySum += coins.front();
    ans++;
    coins.erase(coins.begin());
  }
  printf("%d\n", ans);

  return 0;
}
