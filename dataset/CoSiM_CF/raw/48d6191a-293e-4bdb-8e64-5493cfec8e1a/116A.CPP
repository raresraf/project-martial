// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int x, y, curr = 0, capacity = 0;
  for (int i = 0; i < n; i++) {
    cin >> x >> y;
    curr -= x;
    curr += y;
    capacity = max(capacity, curr);
  }
  cout << capacity;
  return 0;
}