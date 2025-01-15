#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, cnt = 0;
  string a;
  cin >> n >> a;
  for (int i = 1; i < n; i++) {
    if (a[i] == a[i - 1]) {
      cnt++;
    }
  }
  cout << cnt;
}
