#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int length = s.length();
    if (length > 10) {
      cout << s[0] << (length - 2) << s[length - 1] << "\n";
    } else {
      cout << s << "\n";
    }
  }
}