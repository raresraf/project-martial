#include <iostream>

using namespace std;

int main() {
  int n;
  string s;

  cin >> n;

  if (!(1 <= n && n <= 100)) {
    return 0;
  } else {
    ;
  }

  while (n--) {
    cin >> s;
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}