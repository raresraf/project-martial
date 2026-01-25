//#include <bits/stdc++.h>
// using namespace std;
#include <iostream>
#include <string>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);

  int n, x = 0;
  std::cin >> n;
  std::string str;

  for (int i = 0; i < n; n--) {
    std::cin >> str, (str[0] == '+') || (str[2] == '+') ? x++ : x--;
  }
  std::cout << x;

  return 0;
}
