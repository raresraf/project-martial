#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  cin >> t;
  while (t--) {
    char ch[100], a, b;
    cin >> ch;
    if (strlen(ch) <= 10) {
      cout << ch << endl;
    } else if (strlen(ch) > 10) {

      cout << ch[0] << strlen(ch) - 2 << ch[strlen(ch) - 1] << endl;
    }
  }
}
