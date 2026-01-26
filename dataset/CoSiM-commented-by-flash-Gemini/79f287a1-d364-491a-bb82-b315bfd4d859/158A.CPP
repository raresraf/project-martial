#include <iostream>
using std::cin;
using std::cout;
int main() {
  int n, k, cur = 0, ct = 0, rest;
  cin >> n >> k;
  for (int i = 0; i < k; ++i, ++ct) {
    cin >> cur;
    if (!cur) {
      cout << ct;
      return 0;
    }
  }
  while (cin >> rest && cur == rest) {
    ++ct;
  }
  cout << ct;
  return 0;
}
