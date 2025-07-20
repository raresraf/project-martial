/*
  Contest : 160
  Problem A : Twins
  Author : Rakesh Kumar
  Date: 27/08/2020
*/

#include <bits/stdc++.h>

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

  int n = 0;
  std::cin >> n;
  int mudra_kosh = 0;
  std::vector<int> paisa;
  while (n--) {
    int c = 0;
    std::cin >> c;
    paisa.emplace_back(c);
    mudra_kosh += c;
  }

  std::sort(paisa.begin(), paisa.end());

  int j = paisa.size() - 1;
  int s = 0;
  while (j >= 0) {
    s += paisa[j];
    if (s > (mudra_kosh >> 1))
      break;
    --j;
  }
  std::cout << paisa.size() - j << std::endl;

  return 0;
}
