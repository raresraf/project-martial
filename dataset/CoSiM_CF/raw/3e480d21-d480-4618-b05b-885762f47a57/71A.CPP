// http://codeforces.com/problemset/problem/71/A

#include <iostream>
using namespace std;

int main() {
  int n, le;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    le = s.length();
    if (le > 10)
      cout << s[0] << le - 2 << s[le - 1] << endl;
    else
      cout << s << endl;
  }
  return 0;
}
