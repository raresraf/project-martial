#include <bits/stdc++.h>
using namespace std;

#define endl "\n"
#define ends return 0;
#define fast                                                                   \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);

// 1st attempt
void solve() {
  string s;
  getline(cin, s);
  int cnt = 0;

  vector<int> v(26, 0);

  for (int i = 0; s[i]; i++) {
    int x = s[i] - 'a';
    if (v[x] == 0) {
      v[x] = 1;
      cnt++;
    }
  }

  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}

// Second attempt(using stl)
void solve2() {
  string s;
  getline(cin, s);
  int size = s.length();
  set<char> S;

  for (int i = 0; i < size; i++) {
    S.insert(s[i]);
  }

  if (S.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
int main() {
  fast;
  solve2();
  ends;
}