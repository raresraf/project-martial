#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, s, i;
  string str;
  vector<string> v;
  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> str;
    v.push_back(str);
  }

  for (i = 0; i < n; i++) {
    if (v[i].size() > 10) {
      s = v[i].size();
      cout << v[i][0] << s - 2 << v[i][s - 1] << endl;
    }

    else {
      cout << v[i] << endl;
    }
  }

  return 0;
}
