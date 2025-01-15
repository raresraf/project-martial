#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int n = s.length(), dem = 0;
  for (int i = 0; i < n - 1; i++) {

    if (s[i] == s[i + 1]) {
      dem++;
      // cout<<dem<<endl;
      if (dem >= 6)
        break;
    } else
      dem = 0;
  }

  if (dem < 6)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}
