#include <iostream>
using namespace std;
int main() {
  int n;
  int res = 0;
  int num, sure;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    num = 0;
    for (int j = 0; j < 3; ++j) {
      cin >> sure;
      num += sure;
    }
    if (num >= 2)
      res += 1;
  }
  cout << res;
}
