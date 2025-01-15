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

int main() {

  string s;
  cin >> s;

  rep(i, 0, s.size()) toLower(s[i]);

  string v = "aoyeui";
  string new_str = "";

  rep(i, 0, s.size()) {
    if (v.find(s[i]) == string::npos) {
      new_str += ('.');
      new_str += s[i];
    }
  }
  cout << new_str << endl;

  return 0;
}