#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

#define sz(c) (int)((c).size())
#define all(c) c.begin(), c.end()
#define rep(i, a, n) for (int i = a; i < n; i++)

typedef long long ll;
typedef vector<int> vi;
typedef vector<bool> vb;

void toLower(char &c) {
  if (c >= 'a' && c <= 'z') {
  } else {
    c = c + 32;
  }
}

void toLower(string &s) {
  rep(i, 0, s.size()) { toLower(s[i]); }
}

int main() {

  string s, t;
  cin >> s >> t;

  toLower(s);
  toLower(t);

  rep(i, 0, s.size()) {
    if (s[i] < t[i]) {
      puts("-1\n");
      return 0;
    }
    if (s[i] > t[i]) {
      puts("1\n");
      return 0;
    }
  }
  puts("0\n");

  return 0;
}