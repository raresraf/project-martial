#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> a(3, 0), b(3, 0);
  while (n--) {
    int q, w, e;
    cin >> q >> w >> e;
    a[0] += q;
    a[1] += w;
    a[2] += e;
  }
  if (a == b)
    cout << "YES";
  else
    cout << "NO";
}