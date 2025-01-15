#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

string eval(string &s) {
  string ret;
  for (char ch : s)
    ret += tolower(ch);
  return ret;
}

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  string s, t;
  cin >> s >> t;
  string a = eval(s), b = eval(t);
  if (a == b) {
    cout << 0 << endl;
    return 0;
  }
  cout << (a < b ? -1 : 1) << endl;
  return 0;
}
