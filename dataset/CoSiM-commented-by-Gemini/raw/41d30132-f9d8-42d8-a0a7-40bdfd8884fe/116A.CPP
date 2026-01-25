#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, a[1001], b[1001], maxim = 0, cap = 0;
  cin >> n;
  for (int i = 0; i < n; i++)
    cin >> a[i] >> b[i];

  for (int i = 0; i < n; i++) {
    cap = b[i] - a[i] + cap;
    maxim = max(maxim, cap);
  }
  cout << maxim;
}