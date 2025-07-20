/*
  Contest: 1221
  Problem C : Perfect team
  Author : Rakesh Kumar
  Date: 10/06/2020
*/

#include <bits/stdc++.h>

using ll = long long int;

int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(0);

  ll q = 0;
  std::cin >> q;
  while (q--) {
    ll c = 0;
    ll m = 0;
    ll x = 0;
    std::cin >> c >> m >> x;
    ll max_teams = (c + m + x) / 3;
    max_teams = std::min(std::min(c, m), max_teams);
    std::cout << max_teams << std::endl;
  }

  return 0;
}
