#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  while (n--) {
    cin >> s;
    if (s.size() > 10)
      cout << s[0] + to_string(s.size() - 2) + s.back() << endl;
    else
      cout << s << endl;
  }
  return 0;
}