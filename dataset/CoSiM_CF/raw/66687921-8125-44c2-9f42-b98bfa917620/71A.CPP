#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    int x = s.length();
    if (x > 10) {
      cout << s[0] << x - 2 << s[x - 1] << endl;
    } else
      cout << s << endl;
  }
}