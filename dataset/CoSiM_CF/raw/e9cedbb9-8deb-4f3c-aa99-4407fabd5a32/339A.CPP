#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[10000];
  int a[100000], c, i, sum;
  while (cin >> s) {
    int c = 0;

    sum = 0;
    for (i = 0; s[i] != '\0'; i++) {
      if (s[i] == '+') {

        a[c++] = sum;
        sum = 0;
      } else {
        int m = s[i] - 48;
        sum = sum * 10 + m;
      }
    }
    a[c++] = sum;
    sort(a, a + c);
    for (i = 0; i < c - 1; i++)
      cout << a[i] << "+";
    cout << a[c - 1];
    cout << endl;
  }
}
