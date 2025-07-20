#include <bits/stdc++.h>
#include <string>

using namespace std;

int main() {
  string g;
  string h;
  string shuf;

  cin >> g >> h >> shuf;

  string t;
  t = g + h;

  sort(t.begin(), t.end());

  sort(shuf.begin(), shuf.end());

  int count = 0;

  int l = shuf.length();
  for (int i = 0; i < l; i++) {
    if (t.length() > shuf.length()) {
      cout << "NO" << endl;
      break;
    } else if (t[i] == shuf[i])
      count++;
    else {
      cout << "NO" << endl;
      break;
    }
  }

  if (count == l)
    cout << "YES" << endl;
}