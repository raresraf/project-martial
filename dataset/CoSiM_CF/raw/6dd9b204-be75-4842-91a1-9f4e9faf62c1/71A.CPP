#include <iostream>
#include <string>
#define endl "\n"
using namespace std;
int main() {
  int n, i(0), l(0);
  cin >> n;
  string s;
  for (i = 0; i < n; i++) {
    cin >> s;
    if (s.length() <= 10)
      cout << s << endl;
    else {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    }
  }
  return 0;
}
