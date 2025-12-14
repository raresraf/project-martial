#include <bits/stdc++.h>
using namespace std;

bool lucky(int x) {
  if (x < 4)
    return false;
  else {
    while (x > 0) {
      if (x % 10 != 4 && x % 10 != 7)
        return false;
      x /= 10;
    }
    return true;
  }
}

void solve() {
  string s;
  cin >> s;
  int dem = 0;
  int n = s.length();
  for (int i = 0; i < n; i++) {
    if (s[i] == '4' || s[i] == '7')
      dem++;
    // else
    // {
    //   cout<<"NO";
    //  exit(0);
    //}
  }
  if (lucky(dem))
    cout << "YES";
  else
    cout << "NO";
}
int main() {
  solve();

  return 0;
}
