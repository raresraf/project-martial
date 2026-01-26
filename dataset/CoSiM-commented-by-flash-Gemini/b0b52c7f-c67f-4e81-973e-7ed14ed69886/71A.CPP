#include <iostream>

using namespace std;

int main() {
  int n, m, pos = 1;
  cin >> n >> m;

  int64_t res = 0;
  while (m--) {
    int num;
    cin >> num;
    if ((num <= n) && num > pos) {
      res += num - pos;
      pos = num;
    } else if (num < pos) {
      res += (n - pos) + num;
      pos = num;
    }
  }

  cout << res;

  return 0;
}
