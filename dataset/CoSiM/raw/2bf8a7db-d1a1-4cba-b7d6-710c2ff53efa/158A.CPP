#include <iostream>
using namespace std;

int main() {
  int n, k, count;
  cin >> n >> k;
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  count = k;

  for (int i = k - 1; i >= 0; i--) {
    if (a[i] <= 0)
      count--;
  }

  if (count < k) {
    cout << count;
    exit(0);
  }

  for (int i = k; i < n; i++) {
    if (a[i] != a[k - 1])
      break;
    else
      count++;
  }

  cout << count;
  return 0;
}
