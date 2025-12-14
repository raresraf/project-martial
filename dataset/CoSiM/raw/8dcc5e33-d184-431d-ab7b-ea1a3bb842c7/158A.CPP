#include <iostream>
using namespace std;
int main() {
  int k, n, num, a, cnt = 0;
  cin >> n >> k;
  for (int i = 1; i <= n; i++) {
    cin >> a;
    if (i <= k && a > 0)
      cnt++;
    if (i == k) {
      if (a > 0)
        num = a;
    }
    if (i > k && num == a)
      cnt++;
  }
  cout << cnt;
}