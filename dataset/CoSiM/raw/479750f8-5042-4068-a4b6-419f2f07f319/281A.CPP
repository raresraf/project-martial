#include <iostream>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  string s;
  cin >> s;

  for (int i = 0; i < s.length(); i++) {
    if (i == 0) {
      cout << char(toupper(s[i]));
    } else {
      cout << s[i];
    }
  }

  cout << endl;

  return 0;
}
