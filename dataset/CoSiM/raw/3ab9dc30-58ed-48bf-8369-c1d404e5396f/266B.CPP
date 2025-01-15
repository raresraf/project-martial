#include <iostream>

using namespace std;

int main() {
  int n;
  int m;
  string s;
  cin >> n >> m;

  cin >> s;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < s.size(); j++) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j++;
      }
    }
  }
  for (int i = 0; i < n; i++) {
    cout << s[i];
  }
  return 0;
}
