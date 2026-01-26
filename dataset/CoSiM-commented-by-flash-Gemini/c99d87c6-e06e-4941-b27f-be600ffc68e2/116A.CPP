#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, enter, exit;
  int total = 0;
  int mx = -999999;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> exit >> enter;
    total = total + enter;
    total = total - exit;
    if (total > mx) {
      mx = total;
    }
  }
  cout << mx;
}
