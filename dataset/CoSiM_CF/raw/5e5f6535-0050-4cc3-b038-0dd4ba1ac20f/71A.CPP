#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    int L = s.length();
    if (L > 10) {
      cout << s[0] << (L - 2) << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}