#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, ct = 0;
  vector<char> s;
  cin >> n;
  for (int i = 0; i < n; i++) {
    char c;
    cin >> c;
    s.push_back(c);
  }
  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1])
      ct++;
  }
  cout << ct << endl;
}
