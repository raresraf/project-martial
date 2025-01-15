#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  char t;
  string s;
  cin >> n;
  t = getchar();
  while (n--) {
    s = "";
    getline(cin, s);
    if (s.length() >= 11)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl;
  }
  return 0;
}
