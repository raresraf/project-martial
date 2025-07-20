#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int i, j, k, l;
  string s;
  while (cin >> l) {
    cin >> s;
    k = 0;
    for (i = 0; i < l; i++) {
      if (s[i] == s[i + 1]) {
        k++;
      }
    }
    cout << k << endl;
  }
  return 0;
}
