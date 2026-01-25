#include <algorithm>
#include <iostream>
using namespace std;
int a[111];
int main() {
  int n, sum = 0, cnt = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n);
  sum /= 2;
  for (int i = n - 1; sum >= 0; --i) {
    sum -= a[i];
    ++cnt;
  }
  cout << cnt;
}
