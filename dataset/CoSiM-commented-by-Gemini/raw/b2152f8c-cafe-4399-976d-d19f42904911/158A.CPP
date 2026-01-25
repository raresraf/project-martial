#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, j = 0, temp;
  cin >> n >> k;
  vector<int> a;
  for (int i = 0; i < n; ++i) {
    cin >> temp;
    a.push_back(temp);
  }
  int x = a[k - 1];
  for (int i = 0; i < n; ++i) {
    if ((a[i] >= x) && a[i] > 0) {
      j++;
    }
  }
  cout << j;
  return 0;
}