#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, k;
  cin >> n >> k;

  int tmp;
  vector<int> a;
  for (int i = 0; i < n; ++i) {
    cin >> tmp;
    a.push_back(tmp);
  }

  int s = a[k - 1];

  int r = 0;
  for (int i = 0; i < n; ++i) {
    if (a[i] >= s && a[i] > 0)
      r++;
  }

  cout << r << endl;

  return 0;
}
