// link : https://codeforces.com/problemset/problem/266/A

#include <bits/stdc++.h>

using namespace std;

string s;

int a, b, c;

vector<char> v;

main() {

  cin >> a;

  for (int i = 0; i < a; i++) {
    char r;
    cin >> r;
    v.push_back(r);
  }

  b = v.size();

  for (int i = 0; i < b - 1; i++) {

    if (v[i] == v[i + 1]) {
      v.erase(v.begin() + i);
      i--;
      b--;
      c++;
    }
  }

  cout << c;
}
