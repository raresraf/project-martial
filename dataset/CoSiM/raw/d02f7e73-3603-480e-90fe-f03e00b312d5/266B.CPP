#include <iostream>

using namespace std;

int main() {
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;

  for (int i = 0; i < t; i++) {
    for (int j = 1; j < n; j++) {
      if (s[j] == 'G' && s[j - 1] == 'B') {
        s[j] = 'B';
        s[j - 1] = 'G';
        j++;
      }
    }
  }

  cout << s;
  return 0;
}
