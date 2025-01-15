#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, sf = "";
  cin >> s;

  sort(s.begin(), s.end());

  int aux = 0;
  for (int i = s.size() / 2; i < s.size(); ++i) {
    sf += s[i];
    if (i < s.size() - 1)
      sf += "+";
    aux++;
  }

  cout << sf << endl;
}

// codeforces.com/problemset/problem/339/A
