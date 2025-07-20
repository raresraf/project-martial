#include <bits/stdc++.h>
using namespace std;

int main() {
  // freopen("file.text", "r", stdin);
  int n, t;
  cin >> n >> t;

  string ch;
  cin >> ch;

  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (ch[j] == 'B' && ch[j + 1] == 'G') {
        char temp;
        temp = ch[j];
        ch[j] = ch[j + 1];
        ch[j + 1] = temp;
        j++;
      }
    }
  }

  cout << ch << endl;

  return 0;
}
