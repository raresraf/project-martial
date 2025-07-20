/*
  Contest : 158
  Problem A : Next Round
  Author : Rakesh Kumar
  Date: 25/08/2020
*/

#include <bits/stdc++.h>

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

  int n = 0, k = 0;
  std::cin >> n >> k;
  int samman = 0;
  int kth = 0;
  for (int i = 1; i <= n; ++i) {
    int score = 0;
    std::cin >> score;
    if (i == k)
      kth = score;
    if (score && i <= k)
      ++samman;
    if (score && i > k && kth == score)
      ++samman;
  }
  std::cout << samman << std::endl;

  return 0;
}
