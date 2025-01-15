#include <iostream>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  char ch;
  while (t) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        ch = s[i];
        s[i] = s[i + 1];
        s[i + 1] = ch;
        i++;
      }
    }
    t--;
  }
  cout << s;
  return 0;
}
