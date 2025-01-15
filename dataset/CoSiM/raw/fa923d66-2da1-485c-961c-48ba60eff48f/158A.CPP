#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  int count = 0;
  int limit = 0;
  cin >> n >> k;
  for (int i = 1; i <= n; ++i) {
    int a;
    cin >> a;

    if (i == k) {
      limit = a;
    }

    if (i <= k && a > 0) {
      count++;
    }

    if (i > k && a > 0 && a == limit) {
      count++;
    }
  }
  cout << count;
}
