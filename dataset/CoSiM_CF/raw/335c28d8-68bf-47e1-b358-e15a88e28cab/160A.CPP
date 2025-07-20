#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sum = sum / 2;
  sort(a, a + n);
  int sum1 = 0, cnt = 0;
  for (int i = n - 1; i >= 0; i--) {
    sum1 += a[i];
    cnt++;
    if (sum1 > sum) {
      break;
    }
  }
  cout << cnt;
}
