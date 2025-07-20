#include <bits/stdc++.h>
#define _                                                                      \
  ios_base ::sync_with_stdio(0);                                               \
  cin.tie(0);
#define endl "\n";
using namespace std;

int main() {
  int n, total = 0, myTotal = 0;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    total += a[i];
  }
  sort(a.begin(), a.end());
  int remTotal = 0, c = 0;
  for (int i = n - 1; i >= 0; i--) {
    if (myTotal > remTotal)
      break;

    myTotal += a[i];
    remTotal = total - myTotal;
    c++;
  }
  cout << c << endl;
  return 0;
}
