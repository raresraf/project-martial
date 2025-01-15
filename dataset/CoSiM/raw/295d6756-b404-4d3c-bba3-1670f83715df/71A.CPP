// Strings.

#include <iostream>

using namespace std;

int main(void) {
  int n;
  cin >> n;
  string str;
  for (int i = 0; i < n; i++) {
    cin >> str;
    int l = str.length();
    if (l > 10) {
      cout << str[0] << l - 2 << str[l - 1] << endl;
    } else {
      cout << str << endl;
    }
  }

  return 0;
}