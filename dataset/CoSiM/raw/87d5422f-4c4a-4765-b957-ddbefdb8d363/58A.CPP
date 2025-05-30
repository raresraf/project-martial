#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> p) {
  os << p.first << " " << p.second;
  return os;
}
template <typename T> ostream &operator<<(ostream &os, vector<T> &v) {
  for (T i : v)
    os << i << ", ";
  return os;
}
template <typename T> ostream &operator<<(ostream &os, set<T> s) {
  for (T i : s)
    os << i << ", ";
  return os;
}
template <typename T1, typename T2>
ostream &operator<<(ostream &os, map<T1, T2> m) {
  for (pair<T1, T2> i : m)
    os << i << endl;
  return os;
}

int main() {
  string s;
  cin >> s;
  string hello = "hello";
  int cidx = 0;
  for (int i = 0; i < (int)s.length() && cidx < 5; i++) {
    if (s[i] == hello[cidx]) {
      cidx++;
    }
  }
  if (cidx >= 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
