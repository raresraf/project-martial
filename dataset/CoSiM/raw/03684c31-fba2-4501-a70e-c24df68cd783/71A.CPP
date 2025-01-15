#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    int size = s.length();
    if (size > 10) {
      cout << s[0] << (size - 2) << s[size - 1] << endl;
    } else {
      cout << s << endl;
      ;
    }
  }
}
