#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ans = 0;
  cin >> n;
  vector<string> v(n);
  for (int i = 0; i < n; i++) {
    cin >> v[i];
    if ((v[i] == "X++") || (v[i] == "++X")) {
      ans++;
    } else {
      ans--;
    }
  }
  cout << ans << endl;
}
