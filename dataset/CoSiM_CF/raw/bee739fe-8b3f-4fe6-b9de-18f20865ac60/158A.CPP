#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, a[1001];
  int counter = 0;
  cin >> n >> k;

  for (int i = 0; i < n; ++i) {
    cin >> a[i];
  }

  int benchmark = a[k - 1];

  for (int i = 0; i < n; ++i) {
    if (a[i] >= benchmark && a[i] > 0) {
      ++counter;
    }
  }

  cout << counter;
}
