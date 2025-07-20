#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;

  cin >> n >> k;
  int mas[n];

  for (int i(0); i < n; i++) {
    cin >> mas[i];
  }
  int count = 0;
  for (int i(0); i < n; i++) {
    if (mas[i] >= mas[k] && mas[i] > 0) {
      count++;
    }
  }
  cout << count << '\n';
  return 0;
}
