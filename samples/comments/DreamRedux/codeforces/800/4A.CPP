//#include <bits/stdc++.h>
// using namespace std;
#include <iostream>
#include <string>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);

  int w;
  std::cin >> w;
  /**if((w % 2 == 0) && (w >= 4)) {
      std::cout << "YES";
  }
  else {
      std::cout << "NO";
  } */
  std::string ans;

  ans = (w % 2 == 0) && (w >= 4) ? "YES" : "NO";
  std::cout << ans;

  return 0;
}
