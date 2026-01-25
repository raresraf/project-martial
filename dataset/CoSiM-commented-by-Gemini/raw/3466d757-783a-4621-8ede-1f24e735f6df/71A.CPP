#include <iostream>

using namespace std;

int main() {
  int t;
  cin >> t;

  string s;

  while (t--) {
    cin >> s;

    int len = s.length();

    if (len <= 10) {
      cout << s << endl;
    } else {
      cout << s[0] << to_string(len - 2) << s[len - 1] << endl;
    }
  }

  return 0;
}