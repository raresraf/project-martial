#include <bits/stdc++.h>
#include <string>

using namespace std;

int main() {

  int n;
  int t;
  char str[100];

  cin >> n >> t >> str;

  while (t--) {
    for (int i = 0; i < n; i++) {
      if (str[i] == 'B' && str[i + 1] == 'G') {
        swap(str[i], str[i + 1]);
        ++i;
      }
    }
  }

  cout << str << endl;
}
