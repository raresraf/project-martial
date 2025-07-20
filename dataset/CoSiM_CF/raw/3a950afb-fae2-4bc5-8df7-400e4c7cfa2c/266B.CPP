#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;

  string s;
  cin >> s;

  bool sw[50];
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n; j++) {
      sw[j] = false;
    }
    for (int j = 1; j < n; j++) {
      if (s[j - 1] == 'B' && s[j] == 'G') {
        sw[j - 1] = true;
      }
    }

    for (int i = 0; i < n; i++) {
      if (sw[i]) {
        swap(s[i], s[i + 1]);
      }
    }
  }

  cout << s << endl;
}
