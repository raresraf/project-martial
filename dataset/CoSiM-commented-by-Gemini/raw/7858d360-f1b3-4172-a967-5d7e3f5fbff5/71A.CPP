#include <iostream>
using namespace std;
int main() {
  string s;
  int n;
  cin >> n;
  int len = 0;
  for (int i = 0; i < n; i++) {
    cin >> s;
    len = s.length();
    if (len <= 10)
      cout << s << endl;
    else {
      cout << s[0] << len - 2 << s[len - 1] << endl;
    }
  }

  return 0;
}
