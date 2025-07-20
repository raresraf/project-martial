#include <bits/stdc++.h>

using namespace std;
int main() {
  int n, total_coin;
  scanf("%d", &n);
  vector<int> coin_value;
  total_coin = 0;
  int you[2] = {0, 0}; // Total Coin - Sum of Coin

  for (int i = 0; i < n; i++) {
    int x;
    scanf("%d", &x);
    coin_value.push_back(x);
    total_coin += x;
  }

  for (int i = 0; i < n - 1; i++) {
    int swap = i;
    for (int j = i + 1; j < n; j++) {
      if (coin_value[swap] < coin_value[j])
        swap = j;
    }
    if (i != swap) {
      int temp = coin_value[i];
      coin_value[i] = coin_value[swap];
      coin_value[swap] = temp;
    }
  }

  for (int i = 0; i < n; i++) {
    you[1] += coin_value[i];
    you[0] += 1;
    total_coin -= coin_value[i];
    if (you[1] > total_coin) {
      break;
    }
  }

  printf("%d\n", you[0]);
}