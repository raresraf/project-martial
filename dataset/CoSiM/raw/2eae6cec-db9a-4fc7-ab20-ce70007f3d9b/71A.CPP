#include <iostream>
using namespace std;

int main() {
  int n;
  int i = 0;
  cin >> n;
  string s;

  // n-- runs until n != 0
  while (n--) {
    cin >> s;

    if (s.length() > 10) {
      cout << s[i] << s.length() - 2 << s[s.length() - 1] << "\n";
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
