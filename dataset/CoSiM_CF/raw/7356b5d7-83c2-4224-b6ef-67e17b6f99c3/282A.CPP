#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  int ans = 0;
  while (n--) {
    cin >> s;
    if (s == "X++" || s == "++X")
      ans++;
    else
      ans--;
  }
  cout << ans << endl;
  return 0;
}