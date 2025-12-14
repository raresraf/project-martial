#include <bits/stdc++.h>
#include <iostream>
#include <string.h>

using namespace std;

int main() {
  vector<int> s;
  string a;
  cin >> a;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == '1')
      s.push_back(1);
    else if (a[i] == '2')
      s.push_back(2);
    else if (a[i] == '3')
      s.push_back(3);
    else
      cout << "";
  }
  sort(s.begin(), s.end());
  int w = 0;
  for (auto it = s.begin(); it != (s.end() - 1); ++it) {
    cout << *it << "+";
    w++;
  }
  cout << s[w];
  return 0;
}
